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
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
    def __init__(self, net):
        super(ResNetWrapper, self).__init__()
        self.net = net
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        for p in self.net.fc.parameters():
            p.requires_grad = True
       
        # dims = [64, 64, 128, 256]
        # self.adapters = []
        # for i in range(len(dims)):
        #     self.adapters.append( 
        #         nn.Sequential(
        #             nn.BatchNorm2d(dims[i]).to('cuda'),
        #             nn.ReLU(inplace=True).to('cuda'),
        #             conv3x3(dims[i], dims[i]//2).to('cuda'),
        #             nn.BatchNorm2d(dims[i]//2).to('cuda'),
        #             nn.ReLU(inplace=True).to('cuda'),
        #             conv3x3(dims[i]//2, dims[i]).to('cuda'),
        #         )
        #     )
        
        # for m in self.adapters:
        #     for p in m.parameters():
        #         p.requries_grad=True

        dims = [64, 64, 128, 256]
        self.shortcuts = {}
        for i, d_in in enumerate(dims[:-1]):            
            for j, d_out in enumerate(dims):                
                if i>=j: continue                
                self.shortcuts[f'{i}_{j}'] = nn.Sequential
                (
                    conv1x1(d_in, d_out),
                    nn.BatchNorm2d(d_out)
                )
        for k in self.shortcuts.keys():
            print(k)
            for p in self.shortcuts[k].parameters():
                p.requries_grad=True
        self.relu = nn.ReLU(inplace=True)
                    
    def forward(self, x, f0_q=None, f1_q=None, f2_q=None, f3_q=None, get_features=False):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        f0 = self.net.maxpool(x)        

        f1 = self.net.layer1(f0)
        f0_1 = self.shortcuts['0_1'](f0)
        f1 = self.relu(f1 + f0_1)        

        f2 = self.net.layer2(f1)
        f0_2 = self.shortcuts['0_2'](f0)
        f1_2 = self.shortcuts['1_2'](f1)
        f2 = self.relu(f2 + f0_2 + f1_2)       

        f3 = self.net.layer3(f2)
        f0_3 = self.shortcuts['0_3'](f0)
        f1_3 = self.shortcuts['1_3'](f1)
        f2_3 = self.shortcuts['2_3'](f2)
        f3 = self.relu(f3 + f0_3 + f1_3 + f2_3) 

        x = self.net.layer4(f3)
        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)
        o = self.net.fc(x)

        if get_features:
            return o, f0, f1, f2, f3
        else:
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
