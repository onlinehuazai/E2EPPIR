import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import VisionTransformer
from loss import TripletLoss, CrossEntropyLabelSmooth, Arcface


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class MyNet(nn.Module):
    def __init__(self, backbone, embedding_dim, representation_dim, num_classes, bnn_neck='True', weight_factor=0.5):
        super(MyNet, self).__init__()
        self.backbone = backbone
        self.representation_dim = representation_dim
        self.bnn_neck = bnn_neck
        self.embedding_dim = embedding_dim
        self.in_planes = embedding_dim
        self.num_classes = num_classes
        if bnn_neck:
            self.fcneck = nn.Linear(self.embedding_dim, self.representation_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.fcneck_bn = nn.BatchNorm1d(self.representation_dim)
            self.fcneck_bn.bias.requires_grad_(False)
            self.fcneck_bn.apply(weights_init_kaiming)
            self.in_planes = self.representation_dim

        self.classifier = Arcface(self.in_planes, self.num_classes, s=10, m=0.2)
        self.classifier.apply(weights_init_classifier)
        self.triple_loss = TripletLoss(0.6)
        self.ce_loss = CrossEntropyLabelSmooth()
        self.weight_factor = weight_factor

    def forward(self, x, label):
        global_feat = self.backbone(x)
        if self.bnn_neck:
            global_feat = self.fcneck(global_feat)
            global_feat = self.fcneck_bn(global_feat)
        if label is not None:
            cls_score = self.classifier(global_feat, label)
            loss = self.triple_loss(global_feat, label) + self.weight_factor * self.ce_loss(cls_score, label)
            return loss  # global feature for triplet loss
        else:
            return F.normalize(global_feat)


# create net
# net = MyNet(backbone=VisionTransformer(192, 128, 8), embedding_dim=768, representation_dim=256, num_classes=100)
# x = torch.rand(5, 3, 192, 128)
# print(net(x))
