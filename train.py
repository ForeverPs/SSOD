import os
import tqdm
import torch
import shutil
import argparse
from eval import *
import numpy as np
from model import SSOD
from torch import optim
import torch.distributed as dist
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from data import get_train_val_dataset, get_imagenet_ood_dataset


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_ids

    if opt.local_rank == 0 and opt.build_tensorboard:
        shutil.rmtree(opt.logdir, True)
        writer = SummaryWriter(logdir=opt.logdir)
        opt.build_tensorboard = False
    
    dist.init_process_group(backend='nccl', init_method=opt.init_method, world_size=opt.n_gpus)

    batch_size = opt.batch_size
    device = torch.device('cuda', opt.local_rank if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))

    # load dataset
    train_set, val_set = get_train_val_dataset(train_num=opt.train_num)
    ood_set = get_imagenet_ood_dataset(ood_type=opt.ood_type)

    # prepare dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=12)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=6)

    ood_sampler = torch.utils.data.distributed.DistributedSampler(ood_set, shuffle=False)
    ood_loader = DataLoader(ood_set, batch_size=batch_size, sampler=ood_sampler, num_workers=6)
        
    model = SSOD(num_classes=opt.num_classes, train_backbone=opt.train_backbone, train_cls=opt.train_cls)
    
    # loading checkpoint on GPU 0
    if opt.local_rank == 0:
        try:
            model.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=False)
        except:
            print('No Checkpoint, training from scratch...')

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[opt.local_rank],
                                                      output_device=opt.local_rank, broadcast_buffers=False,
                                                      find_unused_parameters=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    for epoch in range(opt.epoch):
        train_loader.sampler.set_epoch(epoch)

        # only tqdm in rank 0
        if opt.local_rank == 0:
            data_loader = tqdm.tqdm(train_loader)
        else:
            data_loader = train_loader
        
        train_loss, val_loss = list(), list()
        train_cls_acc, val_cls_acc = list(), list()

        model.train()
        if not opt.train_cls:
            model.module.cls_head.eval()
        # classification training
        for x, y in data_loader:
            x = x.float().to(device)
            y = y.long().to(device)

            _, cls_logits, loss = model.module.loss(x, y, ood_weight=opt.ood_weight, train_cls=opt.train_cls)

            # record accuracy
            cls_acc = ACC(cls_logits, y)
            train_cls_acc.append(cls_acc)
            train_loss.append(loss.item())

            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update learning rate
        scheduler.step()

        # evaluation
        if opt.local_rank == 0 and epoch % opt.eval_interval == 0:
            model.eval()
            # ID inference
            with torch.no_grad():
                for x, y in tqdm.tqdm(val_loader):
                    x = x.float().to(device)
                    y = y.long().to(device)

                    _, cls_logits, loss = model.module.loss(x, y, ood_weight=opt.ood_weight, train_cls=opt.train_cls)

                    # record accuracy
                    cls_acc = ACC(cls_logits, y)
                    val_cls_acc.append(cls_acc)
                    val_loss.append(loss.item())

            # OOD inference
            id_ood_conf_msp, id_ood_label, id_ood_post_ssod = list(), list(), list()
            with torch.no_grad():
                # ood loader
                for x in tqdm.tqdm(ood_loader):
                    x = x.float().to(device)
                    max_softmax, pred_label, rectified_p = model.module.ood_infer(x)
                    print('OOD Conf: %.4f' % torch.mean(rectified_p))
                    id_ood_conf_msp.extend(max_softmax.detach().squeeze().cpu().numpy().tolist())
                    id_ood_post_ssod.extend(rectified_p.detach().squeeze().cpu().numpy().tolist())
                    id_ood_label.extend(np.zeros(max_softmax.shape[0]).tolist())
                
                # id loader
                id_count = 0
                for x, _ in tqdm.tqdm(val_loader):
                    x = x.float().to(device)
                    max_softmax, pred_label, rectified_p = model.module.ood_infer(x)
                    print('ID Conf: %.4f' % torch.mean(rectified_p))
                    id_ood_conf_msp.extend(max_softmax.detach().squeeze().cpu().numpy().tolist())
                    id_ood_post_ssod.extend(rectified_p.detach().squeeze().cpu().numpy().tolist())
                    id_ood_label.extend(np.ones(max_softmax.shape[0]).tolist())
                    id_count += 1
                    if id_count >= len(ood_loader):
                        break

            assert len(id_ood_conf_msp) == len(id_ood_post_ssod) == len(id_ood_label)

            # calculate FPR
            FPR_msp = FPR(np.array(id_ood_conf_msp), np.array(id_ood_label), threshold=0.95)
            FPR_ssod = FPR(np.array(id_ood_post_ssod), np.array(id_ood_label), threshold=0.95)

            # calculate AUROC
            AUROC_msp = AUROC(np.array(id_ood_conf_msp), np.array(id_ood_label))
            AUROC_ssod = AUROC(np.array(id_ood_post_ssod), np.array(id_ood_label))

            # calculate training loss
            train_loss = np.mean(train_loss)
            train_cls_acc = np.mean(train_cls_acc)

            # calculate validation loss
            val_loss = np.mean(val_loss)
            val_cls_acc = np.mean(val_cls_acc)

            print('EPOCH : %03d | Train Loss : %.4f | Train Cls Acc : %.4f | Val Loss : %.4f | Val Cls Acc : %.4f | '
                  'FPR95(MSP w/ SSOD) : %.4f | FPR95(SSOD) : %.4f | AUROC(MSP w/ SSOD) : %.4f | AUROC(SSOD) : %.4f'
                % (epoch, train_loss, train_cls_acc, val_loss, val_cls_acc, FPR_msp, FPR_ssod, AUROC_msp, AUROC_ssod))

            if FPR_ssod <= opt.best_metric:
                opt.best_metric = FPR_ssod
                model_name = 'epoch_%d_cls_%.4f_fpr95_ssod_%.4f_fpr95_msp_%.4f_auroc_ssod_%.4f_auroc_msp_%.4f.pth' % (epoch, val_cls_acc, FPR_ssod, FPR_msp, AUROC_ssod, AUROC_msp)
                os.makedirs(opt.save_path, exist_ok=True)
                torch.save(model.module.state_dict(), '%s/%s' % (opt.save_path, model_name))

            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/train_cls_acc', train_cls_acc, epoch)

            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/val_cls_acc', val_cls_acc, epoch)

            writer.add_scalar('FPR95/MSP', FPR_msp, epoch)
            writer.add_scalar('FPR95/SSOD', FPR_ssod, epoch)

            writer.add_scalar('AUROC/MSP', AUROC_msp, epoch)
            writer.add_scalar('AUROC/SSOD', AUROC_ssod, epoch)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simplest SSOD')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--init_method', default='env://')
    parser.add_argument('--n_gpus', type=int, default=8)
    parser.add_argument('--device_ids', type=str, default='0,1,2,3,4,5,6,7')

    parser.add_argument('--build_tensorboard', type=bool, default=True)
    parser.add_argument('--best_metric', type=float, default=0.7)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--ood_weight', type=float, default=0.1)
    parser.add_argument('--train_cls', type=bool, default=False)
    parser.add_argument('--train_backbone', type=bool, default=False)
    parser.add_argument('--train_num', type=int, default=10000)

    parser.add_argument('--ood_type', type=str, default='Places')
    parser.add_argument('--logdir', type=str, default='./tensorboard/ssod/ImageNet/Places')
    parser.add_argument('--save_path', type=str, default='./saved_models/ssod')
    parser.add_argument('--checkpoint', type=str, default=None)

    opt = parser.parse_args()
    if opt.local_rank == 0:
        print('opt:', opt)

    main(opt)

# if address already in use, you can use another random master_port
# python3 -m torch.distributed.launch --master_port 9998 --nproc_per_node=8 train.py --n_gpus=8