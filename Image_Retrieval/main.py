import torch
from train import train
from inferrence import test
from model.net import MyNet
from model.backbone import VisionTransformer
from dataset import E2E_Dataset, build_transforms, blockPermutationTransform, RandomErasing
from utils import get_cosine_schedule_with_warmup, RandomIdentitySampler, BaseDataset
import os
import glob
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import re


parser = argparse.ArgumentParser(description="loss parameters")
parser.add_argument("-m", "--m", type=float, default=0.3, help="help of param_m")
parser.add_argument("-s", "--s", type=int, default=32, help="help of param_s")
parser.add_argument("-w", "--weight_factor", type=float, default=0.5, help="help of param_w")
parser.add_argument("-p", "--path", type=str, default='data/cipherimages_Corel10K_8_364', help="help of param_p")
parser.add_argument("-c", "--cuda", type=str, default='1', help="help of param_c")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

# 数据集路径
train_dir = args.path
path_list = glob.glob(os.path.join(train_dir,'*.jpg'))
path_list.sort()
# 划分数据集 Corel10K
labels = [int(path.split('/')[-1].split('.')[0].split('_')[0]) for path in path_list]
seed = 2020
train_list, test_list = train_test_split(path_list, test_size=0.3, stratify=labels, random_state=seed)
train_labels = [int(path.split('/')[-1].split('.')[0].split('_')[0]) for path in train_list]


# labels = [int(re.split(r'(\d+)', path.split('/')[-1])[1]) for path in path_list]
# seed = 2020
# train_list = path_list[:7140]
# test_list = path_list[7140:]
# # train_list, test_list = train_test_split(path_list, test_size=0.3, stratify=labels, random_state=seed)
# train_labels = [int(re.split(r'(\d+)', path.split('/')[-1])[1]) // 4 for path in train_list] 



print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(test_list)}")

# 参数
batch_size = 20
epochs = 200
lr = 6e-3
weight_decay = 1e-5

train_loader = DataLoader(dataset=E2E_Dataset(train_list, transform=build_transforms(True)),
                          batch_size=batch_size, sampler=RandomIdentitySampler(BaseDataset(train_list, train_labels), batch_size, 4))
test_loader = DataLoader(dataset=E2E_Dataset(test_list, transform=build_transforms(False)),
                          batch_size=batch_size, shuffle=False)

model = MyNet(backbone=VisionTransformer(img_dim1=192,img_dim2=192, patch_dim=8, num_layers=5), embedding_dim=768, representation_dim=256, num_classes=100, s=args.s, m=args.m, weight_factor=args.weight_factor).cuda()
print(model)

# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
epoch_start = 1
best_mAP = 0
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=10, num_training_steps=epochs)
# training loop
for epoch in range(epoch_start, 100):
    train_loss = train(model, train_loader, optimizer, epoch, scheduler, epochs)
    if epoch > 60:
        mAP = test(model, test_loader)
        if best_mAP < mAP:
            best_mAP = mAP
            torch.save(model.state_dict(),'model_last_7_256.pth')
print('best_mAP: ', best_mAP)
