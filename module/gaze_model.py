import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.nn.init import normal, constant
import math


__all__ = ['ResNet', 'resnet18', 'GazeLSTM', 'PinBallLoss']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 1000)
        self.fc2 = nn.Linear(1000, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feat_D = self.layer2(x)
        x = self.layer3(feat_D)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

class GazeLSTM(nn.Module):
    def __init__(self):
        super(GazeLSTM, self).__init__()
        self.img_feature_dim = 256
        self.base_model = resnet18(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2, batch_first=True)
        self.last_layer = nn.Linear(2 * self.img_feature_dim, 3)

    def forward(self, input):
        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))

        base_out = base_out.view(input.size(0), 7, self.img_feature_dim)

        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:, 3, :]
        output = self.last_layer(lstm_out).view(-1, 3)

        angular_output = output[:, :2]
        angular_output[:, 0:1] = math.pi * nn.Tanh()(angular_output[:, 0:1])
        angular_output[:, 1:2] = (math.pi / 2) * nn.Tanh()(angular_output[:, 1:2])

        var = math.pi * nn.Sigmoid()(output[:, 2:3])
        var = var.view(-1, 1).expand(var.size(0), 2)

        return angular_output, var


class PinBallLoss(nn.Module):
    def __init__(self):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1 - self.q1

    def forward(self, output_o, target_o, var_o):
        q_10 = target_o - (output_o - var_o)
        q_90 = target_o - (output_o + var_o)
        loss_10 = torch.max(self.q1 * q_10, (self.q1 - 1) * q_10)
        loss_90 = torch.max(self.q9 * q_90, (self.q9 - 1) * q_90)
        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)
        return loss_10 + loss_90