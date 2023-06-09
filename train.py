

import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'

import os
import sys
import json
import time
import torch
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn as nn
from model import LeNet
from utils import train_and_val,plot_acc,plot_loss
import numpy as np




if __name__ == '__main__':


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device ))

    if not os.path.exists('./weight'):
        os.makedirs('./weight')

    BATCH_SIZE = 32

    data_transform = {
        "train": transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
 

    train_dataset = datasets.ImageFolder("./data/train/", transform=data_transform["train"])  # 训练集数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=4)  # 加载数据
    len_train = len(train_dataset)
    val_dataset = datasets.ImageFolder("./data/valid/", transform=data_transform["val"])  # 测试集数据
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=4)  # 加载数据
    len_val = len(val_dataset)

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
  
    # optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-5,amsgrad=True)  # 设置优化器和学习率
    optimizer = optim.Adam(net.parameters(), lr=0.003)  # 设置优化器和学习率
    epoch = 50

    history = train_and_val(epoch, net, train_loader, len_train,val_loader, len_val,loss_function, optimizer,device)


    plot_loss(np.arange(0,epoch), history)
    plot_acc(np.arange(0,epoch), history)


