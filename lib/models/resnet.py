import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False).to('cuda')


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
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetWrapper(nn.Module):
    def __init__(self, nets, num_classes, trainable_params):
        super(ResNetWrapper, self).__init__()
        if len(nets)==1:
            self.net0 = nets[0]
            l = [self.net0]
        else:
            self.net0 = nets[0]
            self.net1 = nets[1]
            self.net2 = nets[2]
            l = [self.net0, self.net1, self.net2]

        for net_curr in l:
            net_curr.fc = torch.nn.Identity().to('cuda')
            net_curr.eval()            
            if 'full' in trainable_params:
                # training the whole backbone
                net_curr.train()
                for p in net_curr.parameters():
                    p.requires_grad = True
            else:
                for p in net_curr.parameters():
                    p.requires_grad = False

            if 'bn' in trainable_params:
                for name, p in net_curr.named_parameters():
                    if 'bn' in name or 'norm' in name:
                        p.requires_grad = True
            net_curr.to('cuda')   
        
        if 'adapter' in trainable_params:
            self._init_adapters()
            for name, p in self.named_parameters():    
                if 'adapter' in name:  
                    p.requires_grad = True

        self.fc = nn.Linear(512 * len(l), num_classes).to('cuda')
        for p in self.fc.parameters():                
            p.requires_grad = True

    def _init_adapters(self):
        p = [64, 64, 128, 256]
        self.adapters0_0 = nn.Sequential(conv1x1(p[0], p[0]), nn.BatchNorm2d(p[0]).to('cuda'))
        self.adapters0_1 = nn.Sequential(conv1x1(p[1], p[1]), nn.BatchNorm2d(p[1]).to('cuda'))
        self.adapters0_2 = nn.Sequential(conv1x1(p[2], p[2]), nn.BatchNorm2d(p[2]).to('cuda'))
        self.adapters0_3 = nn.Sequential(conv1x1(p[3], p[3]), nn.BatchNorm2d(p[3]).to('cuda'))
        if hasattr(self, 'net1'):
            self.adapters1_0 = nn.Sequential(conv1x1(p[0], p[0]), nn.BatchNorm2d(p[0]).to('cuda'))
            self.adapters1_1 = nn.Sequential(conv1x1(p[1], p[1]), nn.BatchNorm2d(p[1]).to('cuda'))
            self.adapters1_2 = nn.Sequential(conv1x1(p[2], p[2]), nn.BatchNorm2d(p[2]).to('cuda'))
            self.adapters1_3 = nn.Sequential(conv1x1(p[3], p[3]), nn.BatchNorm2d(p[3]).to('cuda'))
        if hasattr(self, 'net2'):
            self.adapters2_0 = nn.Sequential(conv1x1(p[0], p[0]), nn.BatchNorm2d(p[0]).to('cuda'))
            self.adapters2_1 = nn.Sequential(conv1x1(p[1], p[1]), nn.BatchNorm2d(p[1]).to('cuda'))
            self.adapters2_2 = nn.Sequential(conv1x1(p[2], p[2]), nn.BatchNorm2d(p[2]).to('cuda'))
            self.adapters2_3 = nn.Sequential(conv1x1(p[3], p[3]), nn.BatchNorm2d(p[3]).to('cuda'))

    def _get_trainable_params_count(self):
        s = 0
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if len(p.shape)==1:
                s += int(p.shape[0])
            else:
                t = 1
                for i in list(p.shape):
                    t *= i
                s += t
        return s

    def _forward_with_adapters(self, net, x, adapters):
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = x + adapters[0](x)

        x = net.layer1(x)
        x = x + adapters[1](x)
        x = net.layer2(x)
        x = x + adapters[2](x)
        x = net.layer3(x)
        x = x + adapters[3](x)
        x = net.layer4(x)

        x = net.avgpool(x)
        x = x.view(x.size(0), -1)
        o = net.fc(x)
        return o

    def forward(self, x, get_features=False):
        if hasattr(self, 'adapters0_0'):
            fs = [self._forward_with_adapters(self.net0, x, [
                self.adapters0_0, self.adapters0_1, self.adapters0_2, self.adapters0_3
            ])]
        else:
            fs = [self.net0(x)]

        if hasattr(self, 'net1'):
            if hasattr(self, 'adapters1_0'):
                fs += [self._forward_with_adapters(self.net1, x, [
                    self.adapters1_0, self.adapters1_1, self.adapters1_2, self.adapters1_3
                ])]
            else:
                fs += [self.net1(x)]

        if hasattr(self, 'net2'):
            if hasattr(self, 'adapters2_0'):
                fs += [self._forward_with_adapters(self.net2, x, [
                    self.adapters2_0, self.adapters2_1, self.adapters2_2, self.adapters2_3
                ])]
            else:
                fs += [self.net2(x)]

        fs = torch.cat(fs, 1)     
        o = self.fc(fs)
        
        if get_features:
            return o, fs
        return o

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride = 2, padding=3, bias=False) # original resnet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1, bias=False)  # resnet in URT
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, get_features=False):
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
        o = self.fc(x)
        # x = F.normalize(x, dim=-1)

        if get_features:
            return o, x

        return o


def resnet18(pretrained=False, ds=None, data_dir=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        if ds:
            model_path = data_dir + 'weights/'+ds+'-net/model_best.pth.tar'
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
