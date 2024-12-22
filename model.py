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
        
        # Initial convolution block (RF: 3x3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.Dropout2d(0.05)
        )
        
        # First block with dilation (RF: 7x7)
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(40, 80, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            DepthwiseSeparableConv(80, 80, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Dropout2d(0.05)
        )
        
        # Second block with increased dilation (RF: 23x23)
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(80, 120, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            DepthwiseSeparableConv(120, 120, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.Dropout2d(0.05)
        )
        
        # Parallel attention branch (SE-like module)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(120, 60, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(60, 120, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Deconvolution block for feature refinement
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(120, 80, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Dropout2d(0.05)
        )
        
        # 1x1 conv for residual connection
        self.residual_conv = nn.Conv2d(120, 80, kernel_size=1)
        
        # Final block with atrous spatial pyramid pooling (ASPP) (RF: 47x47)
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(80, 40, kernel_size=1),
                nn.BatchNorm2d(40),
                nn.ReLU()
            ),
            nn.Sequential(
                DepthwiseSeparableConv(80, 40, kernel_size=3, padding=6, dilation=6),
                nn.BatchNorm2d(40),
                nn.ReLU()
            ),
            nn.Sequential(
                DepthwiseSeparableConv(80, 40, kernel_size=3, padding=12, dilation=12),
                nn.BatchNorm2d(40),
                nn.ReLU()
            ),
            nn.Sequential(
                DepthwiseSeparableConv(80, 40, kernel_size=3, padding=18, dilation=18),
                nn.BatchNorm2d(40),
                nn.ReLU()
            )
        ])
        
        # Final 1x1 convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.05)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)      # 32x32 -> 32x32
        x = self.conv2(x)      # 32x32 -> 32x32
        x = self.conv3(x)      # 32x32 -> 32x32
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        # Residual connection
        identity = self.residual_conv(x)
        x = self.deconv(x)     # 32x32 -> 32x32
        x = x + identity
        
        # ASPP
        aspp_out = []
        for aspp_module in self.aspp:
            aspp_out.append(aspp_module(x))
        x = torch.cat(aspp_out, dim=1)  # Concatenate ASPP outputs
        
        x = self.final_conv(x)  # Combine ASPP features
        x = self.gap(x)        # Global average pooling
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)