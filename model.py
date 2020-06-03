import torch
import torch.nn as nn
import torch.nn.functional as F


from resnet import ResNet, ResNet_CIFAR


class FSConv2d(nn.Module):
    def __init__(self, comp_ratio, in_channels, out_channels, kernel_size, **kwargs):
        super(FSConv2d, self).__init__()
        self.comp_ratio = comp_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = None
        self.kwargs = kwargs
        if 'bias' in kwargs:
            if kwargs['bias']:
                self.bias = nn.Parameter(torch.empty(out_channels))
            del kwargs['bias']

        self.L = (in_channels * out_channels * kernel_size * kernel_size) // comp_ratio
        self.k = in_channels * self.kernel_size * self.kernel_size
        self.s = self.L // out_channels
        self.FS = nn.Parameter(torch.empty(self.L))

        self._init()

    def _init(self):
        nn.init.normal_(self.FS)
        if self.bias:
            nn.init.constant_(self.bias, 0)

    def _build_kernel(self):
        """
        Build conv2d kernel from FS with naive approach
        :return weights: output kernel
        """
        padding = torch.zeros(self.k // 2, device=self.FS.device)
        padded_FS = torch.cat((padding, self.FS, padding))

        kernels = []
        for i in range(self.out_channels):
            patch = torch.narrow(padded_FS, 0, i*self.s, self.k)
            kernels.append(patch)
        
        weights = torch.cat(kernels, 0)
        weights = weights.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        return weights

    def forward(self, x):
        weights = self._build_kernel()
        return F.conv2d(x, weights, **self.kwargs)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return FSConv2d(4, in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return FSConv2d(4, in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = conv3x3(1, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 256)
        self.bn3 = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(256, 1024)
        self.linear2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(self.bn2(out)))
        out = F.relu(self.conv3(self.bn3(out)))
        out = self.avgpool(out)
        out = out.squeeze(out)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


def fs_resnet(data='cifar10', **kwargs):
    num_layers = kwargs.get('num_layers')
    if data == 'cifar10':
        if num_layers == 20:
            model = ResNet_CIFAR(BasicBlock, [3, 3, 3], 10)
        elif num_layers == 32:
            model = ResNet_CIFAR(BasicBlock, [5, 5, 5], 10)
        elif num_layers == 44:
            model = ResNet_CIFAR(BasicBlock, [7, 7, 7], 10)
        elif num_layers == 56:
            model = ResNet_CIFAR(BasicBlock, [9, 9, 9], 10)
        elif num_layers == 110:
            model = ResNet_CIFAR(BasicBlock, [18, 18, 18], 10)
        else:
            return None
        
        # change first layer
        model.conv1 = FSConv2d(4, 3, 16, kernel_size=7, stride=2, padding=3,
                            bias=False)

    elif data == 'cifar100':
        if num_layers == 20:
            model = ResNet_CIFAR(BasicBlock, [3, 3, 3], 100)
        elif num_layers == 32:
            model = ResNet_CIFAR(BasicBlock, [5, 5, 5], 100)
        elif num_layers == 44:
            model = ResNet_CIFAR(BasicBlock, [7, 7, 7], 100)
        elif num_layers == 56:
            model = ResNet_CIFAR(BasicBlock, [9, 9, 9], 100)
        elif num_layers == 110:
            model = ResNet_CIFAR(BasicBlock, [18, 18, 18], 100)
        else:
            return None

        # change first layer
        model.conv1 = FSConv2d(4, 3, 16, kernel_size=7, stride=2, padding=3,
                            bias=False)

    elif data == 'imagenet':
        if num_layers == 18:
            model = ResNet(BasicBlock, [2, 2, 2, 2], 1000)
        elif num_layers == 34:
            model = ResNet(BasicBlock, [3, 4, 6, 3], 1000)
        # elif num_layers == 50:
        #     return ResNet(Bottleneck, [3, 4, 6, 3], 1000)
        # elif num_layers == 101:
        #     return ResNet(Bottleneck, [3, 4, 23, 3], 1000)
        # elif num_layers == 152:
        #     return ResNet(Bottleneck, [3, 8, 36, 3], 1000)
        else:
            return None
        
        # change first layer
        model.conv1 = FSConv2d(4, 3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)

    else:
        return None

    return model
