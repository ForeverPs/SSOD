import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50


class ResBackbone(nn.Module):
    def __init__(self, depth, train_backbone=True):
        super(ResBackbone, self).__init__()
        if depth == 18:
            model = resnet18(pretrained=True)
        elif depth == 34:
            model = resnet34(pretrained=True)
        elif depth == 50:
            model = resnet50(pretrained=True)
        else:
            print('Undefined Model Architectures.')
        
        self.num_channels = model.fc.in_features
        self.backbone = model

        if not train_backbone:
            # freeze the backbone, only train the classification head
            self.backbone.conv1.requires_grad_(False)
            self.backbone.bn1.requires_grad_(False)
            self.backbone.relu.requires_grad_(False)
            self.backbone.maxpool.requires_grad_(False)

            self.backbone.layer1.requires_grad_(False)
            self.backbone.layer2.requires_grad_(False)
            self.backbone.layer3.requires_grad_(False)
            self.backbone.layer4.requires_grad_(False)


    def forward(self, x):
        # x: batch, 3, w, h
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x


class SSOD(nn.Module):
    def __init__(self, num_classes, latent_dim=256, train_cls=True, train_backbone=True):
        super().__init__()

        # resnet backbone
        self.backbone = ResBackbone(depth=50, train_backbone=train_backbone)

        # ood head
        self.ood_head = nn.Sequential(
            nn.Linear(self.backbone.num_channels, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 2)
            )

        # classification head
        if num_classes == 1000:
            self.cls_head = self.backbone.backbone.fc
            if not train_cls:
                self.cls_head.requires_grad_(False)
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(self.backbone.num_channels, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        
        # feature pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    

    def cls_head_forward(self, feat):
        # feat: batch, channel, h, w
        cls_feat = self.avg_pool(feat).reshape(feat.shape[0], -1)
        cls_logits = self.cls_head(cls_feat)

        mask_feat = rearrange(feat, 'b c h w -> b (h w) c')
        mask_logits = self.cls_head(mask_feat)
        return cls_logits, mask_logits
    

    def ood_head_forward(self, feat):
        # feat: batch, channel, h, w
        split_feat = rearrange(feat, 'b c h w -> b (h w) c')
        split_ood_logits = self.ood_head(split_feat)

        avg_feat = self.avg_pool(feat).reshape(feat.shape[0], -1)
        avg_ood_logits = self.ood_head(avg_feat)
        return split_ood_logits, avg_ood_logits


    # def loss(self, x, y, ood_weight=0.1, train_cls=False, thresh=0.99):
    #     # x: batch, 3, h, w, (float)
    #     # y: batch, (long)
    #     feat, cls_logits, mask_logits, split_ood_logits, avg_ood_logits = self.forward(x)

    #     if train_cls:
    #         cls_loss = nn.CrossEntropyLoss()(cls_logits, y)

    #     # ood: 0, id: 1
    #     # mask_logits: batch, (hw), 1000
    #     # mask_conf: batch, (hw)
    #     # mask_label: batch, (hw)
    #     mask_conf, mask_label = torch.max(nn.Softmax(-1)(mask_logits), dim=-1)
        
    #     # Here we only pick the good ID features, all the left are treated as OOD features
    #     # condition 1: predicted label equals to the target
    #     id_ood_label = (mask_label == y.unsqueeze(-1)).long()
    #     # condition 2: only confidence > 0.99 are treated as ID features
    #     mask_conf_binary = rearrange(mask_conf, 'b hw -> (b hw)') > thresh

    #     # used for training the ood head
    #     id_ood_label = rearrange(id_ood_label, 'b hw -> (b hw)') * mask_conf_binary.long()
    #     split_ood_logits = rearrange(split_ood_logits, 'b hw n -> (b hw) n')

    #     # LWB: loss wise balance
    #     id_loss = nn.CrossEntropyLoss(ignore_index=0)(split_ood_logits, id_ood_label)
    #     ood_loss = nn.CrossEntropyLoss(ignore_index=1)(split_ood_logits, id_ood_label)
    #     id_ood_loss = 0.5 * (id_loss + ood_loss)

    #     if train_cls:
    #         loss = cls_loss + ood_weight * id_ood_loss 
    #     else:
    #         loss = ood_weight * id_ood_loss 
    #     return feat, cls_logits, loss


    def loss(self, x, y, ood_weight=0.1, train_cls=False, thresh=0.99):
        # x: batch, 3, h, w, (float)
        # y: batch, (long)
        feat, cls_logits, mask_logits, split_ood_logits, avg_ood_logits = self.forward(x)

        if train_cls:
            cls_loss = nn.CrossEntropyLoss()(cls_logits, y)

        # ood: 0, id: 1
        # mask_logits: batch, (hw), 1000
        # mask_conf: batch, (hw)
        # mask_label: batch, (hw)
        mask_conf, mask_label = torch.max(nn.Softmax(-1)(mask_logits), dim=-1)
        
        # Here we only pick the good ID features as positive data
        # condition 1: predicted label equals to the target
        id_label = (mask_label == y.unsqueeze(-1)).long()
        # condition 2: only confidence > 0.99 are treated as ID features
        mask_conf_binary = rearrange(mask_conf, 'b hw -> (b hw)') > thresh
        
        # used for training the ood head
        id_label = rearrange(id_label, 'b hw -> (b hw)') * mask_conf_binary.long()
        split_ood_logits = rearrange(split_ood_logits, 'b hw n -> (b hw) n')

        # LWB: loss wise balance
        id_loss = nn.CrossEntropyLoss(ignore_index=0)(split_ood_logits, id_label)
        
        # Here we only pick the hard OOD features as positive data
        # condition 1: only confidence < 0.01 are treated as ID features
        mask_conf_binary = rearrange(mask_conf, 'b hw -> (b hw)') < 1.0 - thresh
    
        # used for training the ood head
        # condition 2: predicted label not equals to the target
        ood_label = (1.0 - id_label).long() * mask_conf_binary.long()
        ood_loss = nn.CrossEntropyLoss(ignore_index=1)(split_ood_logits, ood_label)

        # LWB: loss wise balance
        id_ood_loss = 0.5 * (id_loss + ood_loss)

        if train_cls:
            loss = cls_loss + ood_weight * id_ood_loss 
        else:
            loss = ood_weight * id_ood_loss 
        return feat, cls_logits, loss


    def ood_infer(self, x):
        # x: batch, 3, w, h
        feat = self.backbone(x)
        
        # cls head
        # cls_logits: batch, num_classes
        # mask_logits: batch, (HW), num_classes
        # max_softmax: batch,
        # pred_label: batch,
        # mask_label: batch, (HW)
        cls_logits, mask_logits = self.cls_head_forward(feat)
        cls_conf = nn.Softmax(-1)(cls_logits)
        max_softmax, pred_label = torch.max(cls_conf, dim=-1)

        # ood head
        # split_ood_logits: batch, (HW), 2
        # avg_ood_logits: batch, 2
        # id_conf: batch,
        split_ood_logits, avg_ood_logits = self.ood_head_forward(feat)
        id_conf = torch.nn.Softmax(-1)(avg_ood_logits)[:, 1]
        
        # rectified posterior probability
        rectified_p = max_softmax * id_conf
        return max_softmax, pred_label, rectified_p


    def forward(self, x):
        # x: batch, 3, W, H

        # feat: batch, channel, h, w
        feat = self.backbone(x)

        # ID inference
        # cls_logits: batch, num_classes, using global average pooling
        # mask_logits: batch, (h, w), num_classes, w/o using global average pooling
        cls_logits, mask_logits = self.cls_head_forward(feat)

        # OOD inference
        # split_ood_logits: batch, (h w), 1
        # avg_ood_logits: batch, 1
        split_ood_logits, avg_ood_logits = self.ood_head_forward(feat)
        return feat, cls_logits, mask_logits, split_ood_logits, avg_ood_logits
    
    def extract_id_ood_feature(self, x, y, thresh=0.99):
        # x: batch, 3, h, w, (float)
        # y: batch, (long)
        feat, cls_logits, mask_logits, split_ood_logits, avg_ood_logits = self.forward(x)

        # ood: 0, id: 1
        # mask_logits: batch, (hw), 1000
        # mask_conf: batch, (hw)
        # mask_label: batch, (hw)
        mask_conf, mask_label = torch.max(nn.Softmax(-1)(mask_logits), dim=-1)
        
        id_indices, ood_indices = list(), list()
        if y is not None:
            id_label = (mask_label == y.unsqueeze(-1)).long()
            mask_conf_binary = rearrange(mask_conf, 'b hw -> (b hw)') > thresh
            id_ood_label = rearrange(id_label, 'b hw -> (b hw)') * mask_conf_binary.long()

            id_indices = id_ood_label.nonzero().reshape(-1)
            ood_indices = (1 - id_ood_label).nonzero().reshape(-1)


        # id_ood_label = rearrange(id_label, 'b hw -> (b hw)')
        # ood_indices = (1 - id_ood_label).nonzero().reshape(-1)

        feat = rearrange(feat, 'b c h w -> b c (h w)')
        pool_feat = torch.mean(feat, dim=-1)

        feat = rearrange(feat, 'b c hw -> (b hw) c')
        id_feat, ood_feat = None, None
        if len(id_indices):
            id_feat = feat[id_indices]
        
        if len(ood_indices):
            ood_feat = feat[ood_indices]
        
        return id_feat, ood_feat, pool_feat


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    y = (1000 * torch.rand(1)).long()
    model = SSOD(num_classes=1000)
    # _, cls_logits, loss = model.loss(x, y, ood_weight=0.1, train_cls=True, thresh=0.99)
    # print(loss)
    # print(cls_logits.shape)

    id_feat, ood_feat, pool_feat = model.extract_id_ood_feature(x, y)
    print(id_feat, ood_feat, pool_feat.shape)
