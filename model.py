import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, dilation=dilation
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution (RF: 3x3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Depthwise Separable Conv with stride=2 (RF: 7x7)
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Dilated convolution (RF: 15x15)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Deconvolution layer (Transposed Conv)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Final conv with stride=2 (RF: 47x47)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)      # 32x32 -> 32x32
        x = self.conv2(x)      # 32x32 -> 16x16
        x = self.conv3(x)      # 16x16 -> 16x16
        x = self.deconv(x)     # 16x16 -> 32x32
        x = self.conv5(x)      # 32x32 -> 16x16
        x = self.gap(x)        # 16x16 -> 1x1
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)