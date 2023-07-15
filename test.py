import os
import tqdm
import torch
import argparse
from eval import *
import numpy as np
from model import BayesAug
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from data import get_train_val_dataset, get_imagenet_ood_dataset


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_ids
    
    dist.init_process_group(backend='nccl', init_method=opt.init_method, world_size=opt.n_gpus)

    batch_size = opt.batch_size
    device = torch.device('cuda', opt.local_rank if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))

    # load dataset
    _, val_set = get_train_val_dataset()
    ood_set = get_imagenet_ood_dataset(ood_type=opt.ood_type)

    # prepare dataloader
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=6)

    ood_sampler = torch.utils.data.distributed.DistributedSampler(ood_set, shuffle=False)
    ood_loader = DataLoader(ood_set, batch_size=batch_size, sampler=ood_sampler, num_workers=6)
        
    model = BayesAug(depth=opt.depth, num_classes=opt.num_classes)
    
    # loading checkpoint on GPU 0
    if opt.local_rank == 0:
        try:
            model.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)
        except:
            print('No Checkpoint, training from scratch...')

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[opt.local_rank],
                                                      output_device=opt.local_rank, broadcast_buffers=False,
                                                      find_unused_parameters=True)

    # evaluation
    model.eval()
    val_cls_acc, val_loss = 0, 0
    if opt.local_rank == 0:
        # ID inference
        # with torch.no_grad():
        #     for x, y in tqdm.tqdm(val_loader):
        #         x = x.float().to(device)
        #         y = y.long().to(device)
        #         _, cls_logits, loss = model.module.loss(x, y, ood_weight=opt.ood_weight, get_feat=True, ood_loss=True)

        #         # record accuracy
        #         cls_acc = ACC(cls_logits, y)
        #         val_cls_acc += cls_acc
        #         val_loss += loss.item()

        # OOD inference
        id_ood_conf_msp, id_ood_conf_BayesAug, id_ood_label = list(), list(), list()
        with torch.no_grad():
            # ood loader
            for x in tqdm.tqdm(ood_loader):
                x = x.float().to(device)
                max_softmax, pred_label, rectified_p, id_conf = model.module.ood_infer(x)
                if x.shape[0] > 1:
                    id_ood_conf_msp.extend(max_softmax.detach().squeeze().cpu().numpy().tolist())
                    id_ood_conf_BayesAug.extend(rectified_p.detach().squeeze().cpu().numpy().tolist())
                    id_ood_label.extend(np.zeros(max_softmax.shape[0]).tolist())
                else:
                    id_ood_conf_msp.append(max_softmax.detach().squeeze().cpu().item())
                    id_ood_conf_BayesAug.append(rectified_p.detach().squeeze().cpu().item())
                    id_ood_label.append(0)

                print('OOD Conf:', rectified_p.mean())
            
            # id loader
            for x, _ in tqdm.tqdm(val_loader):
                x = x.float().to(device)
                max_softmax, pred_label, rectified_p, id_conf = model.module.ood_infer(x)
                if x.shape[0] > 1:
                    id_ood_conf_msp.extend(max_softmax.detach().squeeze().cpu().numpy().tolist())
                    id_ood_conf_BayesAug.extend(rectified_p.detach().squeeze().cpu().numpy().tolist())
                    id_ood_label.extend(np.ones(max_softmax.shape[0]).tolist())
                else:
                    id_ood_conf_msp.append(max_softmax.detach().squeeze().cpu().item())
                    id_ood_conf_BayesAug.append(rectified_p.detach().squeeze().cpu().item())
                    id_ood_label.append(1)

                print('ID Conf:', rectified_p.mean())

        assert len(id_ood_conf_msp) == len(id_ood_conf_BayesAug) == len(id_ood_label)

        FPR_msp = FPR(np.array(id_ood_conf_msp), np.array(id_ood_label), threshold=0.95)
        FPR_BayesAug = FPR(np.array(id_ood_conf_BayesAug), np.array(id_ood_label), threshold=0.95)

        AUROC_msp = AUROC(np.array(id_ood_conf_msp), np.array(id_ood_label))
        AUROC_BayesAug = AUROC(np.array(id_ood_conf_BayesAug), np.array(id_ood_label))

        val_loss = val_loss / len(val_loader)
        val_cls_acc = val_cls_acc / len(val_loader)

        print('Dataset : %s | Val Loss : %.4f | Val Cls Acc : %.4f | FPR95(MSP) : %.4f | FPR95(BayesAug) : %.4f | AUROC(MSP) : %.4f | AUROC(BayesAug) : %.4f'
            % (opt.ood_type, val_loss, val_cls_acc, FPR_msp, FPR_BayesAug, AUROC_msp, AUROC_BayesAug))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('BayesAug Evaluation')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--init_method', default='env://')

    parser.add_argument('--n_gpus', type=int, default=7)
    parser.add_argument('--device_ids', type=str, default='1,2,3,4,5,6,7')

    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--ood_weight', type=float, default=0.1)
    # parser.add_argument('--ood_type', type=str, default='iNaturalist')
    # parser.add_argument('--ood_type', type=str, default='SUN')
    # parser.add_argument('--ood_type', type=str, default='Places')
    parser.add_argument('--ood_type', type=str, default='Texture')
    parser.add_argument('--checkpoint', type=str, default='./saved_models/BayesAug_ResNet50/ImageNet/loss_balance_prototype/epoch_32_cls_0.7414_fpr95_BayesAug_0.2946_fpr95_msp_0.4423_auroc_BayesAug_0.9213.pth')

    opt = parser.parse_args()
    if opt.local_rank == 0:
        print('opt:', opt)

    main(opt)

# if address already in use, you can use another random master_port
# python3 -m torch.distributed.launch --master_port 9998 --nproc_per_node=7 test.py --n_gpus=7