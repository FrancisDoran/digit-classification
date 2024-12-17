import torch
from torch import nn
import torch.nn.functional as F

class SAFPooling(nn.Module):
    """
    SAF-Pooling: A pooling mechanism that pools the highest activations 
    and suppresses some randomly to improve robustness.
    """
    def __init__(self, pool_size):
        super(SAFPooling, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        # Max pooling for highest activations
        x_max = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
        # Random dropout of some activations
        mask = torch.bernoulli(torch.full_like(x_max, 0.9))  # Keep 90% activations
        return x_max * mask

class ConvBlock(nn.Module):
    """
    A convolutional block with Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SimpNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(SimpNet, self).__init__()
        self.features = nn.Sequential(
            # Group 1
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64),
            SAFPooling(pool_size=2),  # Output: 64x14x14

            # Group 2
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            SAFPooling(pool_size=2),  # Output: 128x7x7

            # Group 3
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            SAFPooling(pool_size=2)   # Output: 256x3x3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),  # Adjusted input size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        