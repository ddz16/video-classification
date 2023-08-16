import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from model import generate_model
from train import train, val, test
from dataset import Video
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop, ToTensor)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--exp', default='train', type=str, help='[train]')

parser.add_argument('--data', default='data/', type=str, help='dataset file path')
parser.add_argument('--sample_size', default=256, type=int, help='the size of input frame')
parser.add_argument('--downsample_rate', default=2, type=int, help='frames/second, i.e., each second has two frames')
parser.add_argument('--sample_duration', default=16, type=int, help='the length of clip')
parser.add_argument('--n_classes', default=7, type=int, help='the number of classes')

parser.add_argument('--model', default='', type=str, help='checkpoint model file')
parser.add_argument('--output', default='output.json', type=str, help='Output file path')

parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
parser.add_argument('--model_depth', default=10, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')

parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=8, type=int, help='num_workers')
parser.add_argument('--checkpoint_path', default='checkpoints/', type=str)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


model = generate_model(opt)
# print(model)
device = torch.device("cuda:0")
train_epoch = 40


spatial_transform = Compose([
    Resize(opt.sample_size),
    CenterCrop(opt.sample_size),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_dataset = Video(root_path=opt.data, flag='train', spatial_transform=spatial_transform, downsample_rate=opt.downsample_rate, sample_duration=opt.sample_duration)
test_dataset = Video(root_path=opt.data, flag='test', spatial_transform=spatial_transform, downsample_rate=opt.downsample_rate, sample_duration=opt.sample_duration)

print("train dataset length:", len(train_dataset))
print("test dataset length:", len(test_dataset))
# train_class_num = [0,0,0,0,0,0]
# test_class_num = [0,0,0,0,0,0]
# label_dic = {'dive': 0, 'walk': 1, 'observe':2, 'work': 3, 'ascend': 4, 'off': 5}
# for i in range(len(train_dataset)):
#     train_class_num[int(train_dataset[i][1])] += 1
# for i in range(len(test_dataset)):
#     test_class_num[int(test_dataset[i][1])] += 1
# print("the number of sample clips in each class in train dataset:", train_class_num)
# print("the number of sample clips in each class in test dataset:", test_class_num)

# train_size = int(len(mydataset) * 0.8)
# test_size = len(mydataset) - train_size
# print('train size: {}, test_size: {}'.format(train_size, test_size))

# train_dataset, test_dataset = torch.utils.data.random_split(
#     dataset=mydataset,
#     lengths=[train_size, test_size],
#     generator=torch.Generator().manual_seed(0)
# )

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size, 
    shuffle=True, 
    num_workers=opt.num_workers, 
    pin_memory=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batch_size, 
    shuffle=False,
    num_workers=opt.num_workers, 
    pin_memory=False
)

if opt.exp == 'train':
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    train(model, criterion, optimizer, train_loader, test_loader, train_epoch, device, opt)
    


