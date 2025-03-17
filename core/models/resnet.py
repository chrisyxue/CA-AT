import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.classifier import *
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
import torchvision

import pdb

import torch.nn as nn

def disable_bn_track_running_stats(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False
    return model

def able_bn_track_running_stats(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = True
    return model


class Normalization(nn.Module):
    """
    Standardizes the input data.
    Arguments:
        mean (list): mean.
        std (float): standard deviation.
        device (str or torch.device): device to be used.
    Returns:
        (input - mean) / std
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        num_channels = len(mean)
        self.mean = torch.FloatTensor(mean).view(1, num_channels, 1, 1)
        self.sigma = torch.FloatTensor(std).view(1, num_channels, 1, 1)
        self.mean_cuda, self.sigma_cuda = None, None

    def forward(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.sigma_cuda = self.sigma.cuda()
            out = (x - self.mean_cuda) / self.sigma_cuda
        else:
            out = (x - self.mean) / self.sigma
        return out


class BasicBlock(nn.Module):
    """
    Implements a basic block module for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes,track_running_stats=False,affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes,track_running_stats=False,affine=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes,track_running_stats=False,affine=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Implements a basic block module with bottleneck for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes,track_running_stats=False,affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes,track_running_stats=False,affine=True)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes,track_running_stats=False,affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes,track_running_stats=False,affine=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet model
    Arguments:
        block (BasicBlock or Bottleneck): type of basic block to be used.
        num_blocks (list): number of blocks in each sub-module.
        num_classes (int): number of output classes.
        device (torch.device or str): device to work on. 
    """
    def __init__(self, block, num_blocks, num_classes=10, device='cpu',classifier='linear'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if classifier == 'linear':
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        elif classifier == 'cosine':
            self.linear = NormedLinear(512 * block.expansion, num_classes)
        else:
            raise ValueError('No classifier named '+str(classifier))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def encode(self,x,normalize=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if normalize:
            out = out/torch.norm(out,dim=-1)
        return out


def resnet(name, num_classes=10, pretrained=False, device='cpu', classifier='linear'):
    """
    Returns suitable Resnet model from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        pretrained (bool): whether to use a pretrained model.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if name == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, device=device, classifier=classifier)
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18().to(device)
            # model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1, bias=False)
            model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True),strict=False)
            model = disable_bn_track_running_stats(model=model)
            # pdb.set_trace()
            model.fc = nn.Linear(512,num_classes,bias=False)
        return model
    
    elif name == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, device=device, classifier=classifier)
    elif name == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, device=device, classifier=classifier)
    elif name == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, device=device, classifier=classifier)
    
    elif name == 'tiny-renset50':
        model = torchvision.models.get_model('resnet50', weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Identity()
        # model = disable_bn_track_running_stats(model=model)
        return model
    
    elif name == 'tiny-resnet18':
        model = torchvision.models.get_model('resnet18', weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Identity()
        model = disable_bn_track_running_stats(model=model)
        return model
    
    raise ValueError('Only resnet18, resnet34, resnet50 and resnet101 are supported!')
