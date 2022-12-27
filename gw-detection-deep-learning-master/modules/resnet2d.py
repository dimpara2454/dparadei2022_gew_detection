'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3, fully_conv=False, a=1,
                 feature_extraction_only=False):
        super(ResNet, self).__init__()
        self.a = a
        self.in_planes = int(a * 64)
        self.feature_extraction_only = feature_extraction_only

        self.conv1 = nn.Conv2d(in_channels, int(a * 64), kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(a * 64))
        self.layer1 = self._make_layer(block, int(a * 64), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(a * 128), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(a * 256), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(a * 512), num_blocks[3], stride=2)

        if not feature_extraction_only:
            if fully_conv:
                self.linear = nn.Conv2d(int(a * 512) * block.expansion, num_classes, kernel_size=1, stride=1, padding=0)
            else:
                self.linear = nn.Linear(int(a * 512) * block.expansion, num_classes)
        self.fully_conv = fully_conv
        self.hidden = None
        self.hidden_channels = int(a * 256)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def feature_extraction(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self.hidden = out
        out = self.layer4(out)
        return out

    def classify(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))

        if self.fully_conv:
            out = self.linear(out)
            return out
        out = out.view(out.size(0), -1)
        # self.hidden = out
        out = F.softmax(self.linear(out), dim=1)
        return out

    def forward(self, x):
        out = self.feature_extraction(x)

        if not self.feature_extraction_only:
            # out = F.avg_pool2d(out, 4)
            out = self.classify(out)
        return out


def ResNet8(in_channels=3, n_classes=10, fully_conv=False, a=1, fe_only=False):
    return ResNet(BasicBlock, [1, 1, 1, 1], in_channels=in_channels, num_classes=n_classes, fully_conv=fully_conv, a=a, feature_extraction_only=fe_only)


def ResNet18(in_channels=3, n_classes=10, fe_only=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=n_classes, feature_extraction_only=fe_only)


def ResNet34(in_channels=3, n_classes=10, fe_only=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, num_classes=n_classes,
                  feature_extraction_only=fe_only)


def ResNet50(in_channels=3, n_classes=10, fe_only=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=n_classes,
                  feature_extraction_only=fe_only)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


if __name__ == '__main__':
    net = ResNet50(in_channels=2, n_classes=2)
    x = torch.randn(8, 2, 128, 129)
    out = net(x)
    print(out.size())
