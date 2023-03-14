import torch
import torch.nn as nn

'''
This implementation assumes that the input images are RGB images of size 32x32. The first convolutional layer takes in 3 input channels (corresponding to the R, G, and B channels), and outputs 16 feature maps. The _make_layer function creates a residual block with two convolutional layers and a skip connection, which is applied n times for each of the three sets of hidden layers.

The output layer consists of an adaptive average pooling layer that averages the feature maps over the spatial dimensions, followed by a fully connected layer with r units, where r is the number of classes.

You can create an instance of the ResNet model by calling the resnet_6n_2 function with the desired values of n and r. For example:
'''


class MyLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(MyLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [1] * len(x.size())
        shape[1] = x.size(1)
        mean = x.mean(dim=tuple(range(1, len(x.size()))), keepdim=True)
        std = x.std(dim=tuple(range(1, len(x.size()))), keepdim=True)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            x = x * self.weight.reshape(shape) + self.bias.reshape(shape)
        return x















class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MyLayerNorm(num_features = out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MyLayerNorm(num_features = out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, n, r):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = MyLayerNorm(num_features = 16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, n)
        self.layer2 = self.make_layer(32, n, stride=2)
        self.layer3 = self.make_layer(64, n, stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, r)
        

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        out = self.conv(x)
        #print("debug--batch--norm")
        #print("debug--batch--norm", out.shape)
        out = self.bn(out)
        #print("debug--batch--norm", out.shape)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
