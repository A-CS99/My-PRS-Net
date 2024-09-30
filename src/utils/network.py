"""
实现CNN网络结构：
    - 5个3D卷积层
        - 卷积核大小（kernel size）为3
        - 填充（padding）为1
        - 步幅（stride）为1
        - 每层后使用最大值池化（Max Pooling），卷积核大小为2
        - 激活函数使用Leaky ReLU
    - 全连接层
输入：
    32x32x32像素的3D体素数据
输出：
    3个4参数隐式表示的对称平面，3个4参数轴角表示的旋转轴
"""
import torch
import torch.nn as nn

class PRS_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 5个3D卷积层配置
        self.conv1 = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(4, 8, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(8, 16, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(16, 32, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        # 全连接层配置
        self.plane_seqs = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4)),
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4)),
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4))
        ])
        self.quat_seqs = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4)),
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4)),
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4))
        ])

    def conv_layers(self, input):
        '''
        5个3D卷积层实现
        @param input: 输入数据[batch_size, 1, 32, 32, 32]
        @return output: 输出数据[batch_size, 64, 1, 1, 1]
        '''
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.conv5(x)
        return output
    
    def full_connect(self, input):
        '''
        全连接层实现
        @param input: 输入数据[batch_size, 64, 1, 1, 1]
        @return planes: 3个4参数隐式表示的对称平面[batch_size, 3, 4]
        @return quaternions: 3个4参数轴角表示的旋转轴[batch_size, 3, 4]
        '''
        planes = []
        quaternions = []
        for i in range(3):
            x = input.view(input.size(0), -1)
            plane = self.plane_seqs[i](x)
            quaternion = self.quat_seqs[i](x)
            planes.append(plane)
            quaternions.append(quaternion)
        planes = torch.stack(planes, dim=1)
        quaternions = torch.stack(quaternions, dim=1)
        return planes, quaternions

    def forward(self, input):
        # 前向传播
        x = self.conv_layers(input)
        output = self.full_connect(x)
        return output
