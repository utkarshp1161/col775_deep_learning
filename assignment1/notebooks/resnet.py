import torch.nn as nn


class ResNet(nn.Module):
    '''
    This implementation assumes that the input images are RGB images of size 32x32. The first convolutional layer takes in 3 input channels (corresponding to the R, G, and B channels), and outputs 16 feature maps. The _make_layer function creates a residual block with two convolutional layers and a skip connection, which is applied n times for each of the three sets of hidden layers.

    The output layer consists of an adaptive average pooling layer that averages the feature maps over the spatial dimensions, followed by a fully connected layer with r units, where r is the number of classes.

    You can create an instance of the ResNet model by calling the resnet_6n_2 function with the desired values of n and r. For example:
    '''
    def __init__(self, n, r):
        super(ResNet, self).__init__()
        self.n = n
        self.r = r
        
        # first convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # residual blocks
        self.layer1 = self._make_layer(16, 16, 1)
        self.layer2 = self._make_layer(16, 32, 2)
        self.layer3 = self._make_layer(32, 64, 2)
        
        # output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.r)
        
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

