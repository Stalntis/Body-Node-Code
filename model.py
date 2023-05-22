import torch.nn as nn
import torch.nn.functional as F
# from CBAM import CBAMLayer
from SMU import SMU
from CA import CA_Block

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1) # in_channels=3 out_channels=6 kernel=5  16*32*32
        self.SMU = SMU()

        self.pool1 = nn.MaxPool2d(kernel_size = 2 ,stride = 2,padding=0)

        # self.se1 = CBAMLayer(channel=6, reduction=4)
        self.conv2 = nn.Conv2d(16,32,kernel_size=5,stride=1,padding=0) #32*12*12
        self.SMU = SMU()

        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride = 2,padding=0)

        self.ca = CA_Block(channel=32,h =6,w =6)
        # self.eca = eca_block(channel=16)
        self.fc1 = nn.Linear(32*6*6, 120)
        self.SMU = SMU()

        self.fc2 = nn.Linear(120, 84)
        self.SMU = SMU()

        self.fc3 = nn.Linear(84, 25)




# 调用上面定义的函数
    def forward(self, x):


        # x = F.SMU(self.conv1(x))    # input(3, 32, 32) output(6, 32, 32)
        #
        # x = self.pool1(x)       # output(6, 16, 16)
        #
        # x = F.SMU(self.conv2(x))    # output(16, 12, 12)
        #
        # x = self.pool2(x)            # output(16, 6, 6)
        #
        #
        #
        # x = x.view(-1, 16*5*5)       # output(32*5*5)
        #
        # x = F.SMU(self.fc1(x))      # output(120)
        # x = F.SMU(self.fc2(x))      # output(84)
        # x = self.fc3(x)    # output(10)
        x = self.conv1(x)
        x = self.SMU(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.SMU(x)
        x = self.pool2(x)
        x = self.ca(x)
        x = x.view(-1, 32 * 6 * 6)
        x = self.fc1(x)
        x = self.SMU(x)
        x = self.fc2(x)
        x = self.SMU(x)
        x = self.fc3(x)




        return x
