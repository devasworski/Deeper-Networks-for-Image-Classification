# code based on https://gist.github.com/nikogamulin/7774e0e3988305a78fd73e1c4364aded
import torch

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = torch.nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class resnet(torch.nn.Module):
    def __init__(self, num_classes):
        super(resnet, self).__init__()

        self.in_channels = 64
        self.expansion = 4
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers=3, channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers=4, channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers=6, channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers=3, channels=512, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, channels, stride):
        layers = []
        identity_downsample = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, channels*self.expansion, kernel_size=1, stride=stride),
            torch.nn.BatchNorm2d(channels*self.expansion)
            )
        layers.append(Block(self.in_channels, channels, identity_downsample, stride))
        self.in_channels = channels * self.expansion
        for _ in range(num_layers-1):
            layers.append(Block(self.in_channels, channels))
        return torch.nn.Sequential(*layers)
