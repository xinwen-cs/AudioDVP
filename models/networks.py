import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

from utils import util
from renderer.face_model import FaceModel


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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

        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class ResnetFaceModelOptimizer(nn.Module):
    def __init__(self, opt):
        super(ResnetFaceModelOptimizer, self).__init__()
        self.pretrained_model = ResNet()
        self.fc = nn.Linear(2048, 97)  # 257 - 160 = 97
        self.face_model = FaceModel(data_path=opt.matlab_data_path, batch_size=opt.batch_size)

        self.init_weights(opt.pretrained_model_path)

        self.alpha = nn.Parameter(torch.zeros((1, 80, 1), device=opt.device))  # shared for all samples
        self.beta = nn.Parameter(torch.zeros((1, 80, 1), device=opt.device))  # shared for all samples

        self.to(opt.device)

    def init_weights(self, pretrained_model_path):
        util.load_state_dict(self.pretrained_model, pretrained_model_path)

        torch.nn.init.zeros_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        coef = self.fc(self.pretrained_model(x)).unsqueeze(-1)

        delta = coef[:, :64]
        rotation = coef[:, 91:94]
        translation = coef[:, 94:]
        gamma = coef[:, 64:91]

        render, mask, screen_vertices = self.face_model(self.alpha, delta, self.beta, rotation, translation, gamma)

        return self.alpha, delta, self.beta, gamma, rotation, translation, render, mask, screen_vertices


class CoefficientRegularization(nn.Module):
    def __init__(self):
        super(CoefficientRegularization, self).__init__()

    def forward(self, input):
        return torch.sum(input**2)


class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)


class LandmarkLoss(nn.Module):
    def __init__(self, opt):
        super(LandmarkLoss, self).__init__()

        self.device = opt.device

        self.landmark_weight = torch.tensor([[
                1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                1.0,  1.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        ]], device=self.device).unsqueeze(-1)

    def forward(self, landmark, landmark_gt):
        landmark_loss = (landmark - landmark_gt) ** 2
        landmark_loss = torch.sum(self.landmark_weight * landmark_loss) / 68.0

        return landmark_loss


class AudioExpressionModule(nn.Module):
    def __init__(self, opt):
        super(AudioExpressionModule, self).__init__()
        self.opt = opt
        self.conv1 = nn.Conv1d(opt.Nw, 5, 3)
        self.conv2 = nn.Conv1d(5, 3, 3)
        self.conv3 = nn.Conv1d(3, 1, 3)
        self.fc = nn.Linear(250, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.unsqueeze(-1)


class MouthMask(nn.Module):
    def __init__(self, opt):
        super(MouthMask, self).__init__()
        self.face_model = FaceModel(data_path=opt.matlab_data_path, batch_size=1)
        self.tensor2pil = transforms.ToPILImage()

    def forward(self, alpha, delta, beta, gamma, rotation, translation):
        delta = delta.clone()

        delta[0, 0, 0] = -8.0
        _, open_mask, _ = self.face_model(alpha, delta, beta, rotation, translation, gamma, lower=True)

        delta[:, :, :] = 0.0
        _, close_mask, _ = self.face_model(alpha, delta, beta, rotation, translation, gamma, lower=True)

        mouth_mask = torch.clamp(open_mask + close_mask, min=0.0, max=1.0)

        return mouth_mask
