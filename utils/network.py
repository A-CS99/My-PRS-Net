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
import torch.nn as nn
import torch.nn.functional as F

class PRS_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 5个3D卷积层配置
        self.conv1 = nn.Conv3d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv3d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv3d(32, 64, 3, padding=1)
        # 每层后使用最大值池化
        self.max_pool = nn.MaxPool3d(2, 2)
        # 全连接层配置
        self.fc1 = nn.ModuleList([nn.Linear(64, 32), nn.Linear(32, 16), nn.Linear(16, 4)])
        self.fc2 = nn.ModuleList([nn.Linear(64, 32), nn.Linear(32, 16), nn.Linear(16, 4)])
        self.fc3 = nn.ModuleList([nn.Linear(64, 32), nn.Linear(32, 16), nn.Linear(16, 4)])

    def conv_layers(self, input):
        # 5个3D卷积层实现
        x = self.max_pool(F.leaky_relu(self.conv1(input)))
        x = self.max_pool(F.leaky_relu(self.conv2(x)))
        x = self.max_pool(F.leaky_relu(self.conv3(x)))
        x = self.max_pool(F.leaky_relu(self.conv4(x)))
        output = self.max_pool(F.leaky_relu(self.conv5(x)))
        return output
    
    def full_connect(self, input):
        # 全连接层实现
        outputs = []
        x = input.view(-1, 64)
        for fc in self.fc1:
            x = F.leaky_relu(fc(x))
        for fc in self.fc2:
            x = F.leaky_relu(fc(x))
        for fc in self.fc3:
            x = fc(x)
            outputs.append(x)
        return outputs

    def forward(self, input):
        # 前向传播
        x = self.conv_layers(input)
        output = self.full_connect(x)
        return output

# 实例化网络
net = PRS_Net()
print(net)