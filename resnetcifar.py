'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)
    


class BasicBlock(nn.Module):
    expansion = 1

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
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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


                        
class ResNetCifar_SD(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCifar_SD, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)

        self.scala = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            # use (4,4) for cifar-10 and 100 and (8,8) for tiny_imagenet
            nn.AvgPool2d(4, 4),  
        )
        
        self.attention = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, weights = None, get_feat=None, SD = None):
        # See note [TorchScript super()]
        if weights == None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            
    
            feat1 = self.layer1(x)
            feat2 = self.layer2(feat1)
            
            feat3 = self.layer3(feat2)
            feat4 = self.layer4(feat3)
            
            if SD != None:
                atten = self.attention(feat2)
                feat2 = atten*feat2
                SD_x = self.scala(feat2)
                SD_feat = torch.flatten(SD_x, 1)
                SD_x = self.fc2(SD_feat)
            
            x = self.avgpool(feat4)
            feat = torch.flatten(x, 1)
            x = self.fc(feat)
            
            
            
            if get_feat == None and SD ==None:
                return x
            elif get_feat != None and SD ==None:
                return x, feat
            elif get_feat != None and SD != None:
                return x, SD_x, feat 
            elif get_feat == None and SD != None:
                return x, SD_x 
            
        else:     
            x = F.conv2d(x, weights['conv1.weight'], bias=None, stride=1, padding=1)
            x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, weights['bn1.weight'], weights['bn1.bias'],training=True)            
            x = F.relu(x, inplace=True)
            #layer 1
            for i in range(2):
                residual = x
                out = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                 weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i],training=True)      
                out = F.relu(out, inplace=True)
                out = F.conv2d(out, weights['layer1.%d.conv2.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                 weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i],training=True)   
                out = F.relu(out, inplace=True)                         
                x = out + residual     
                x = F.relu(x, inplace=True)
                feat1 = x

            #layer 2
            for i in range(2):
                residual = x
                if i == 0:
                    out = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], bias=None, stride=2, padding=1)
                else:
                    out = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                 weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i],training=True)     
                out = F.relu(out, inplace=True)
                out = F.conv2d(out, weights['layer2.%d.conv2.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                 weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i],training=True)    
                if i==0:
                    residual = F.conv2d(x, weights['layer2.%d.downsample.0.weight'%i], bias=None, stride=2)  
                    residual = F.batch_norm(residual, self.layer2[i].downsample[1].running_mean, self.layer2[i].downsample[1].running_var, 
                                 weights['layer2.%d.downsample.1.weight'%i], weights['layer2.%d.downsample.1.bias'%i],training=True)  
                x = out + residual  
                x = F.relu(x, inplace=True)
            feat2 = x
               
            #layer 3
            for i in range(2):
                residual = x
                if i == 0:
                    out = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], bias=None, stride=2, padding=1)
                else:
                    out = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                 weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i],training=True)     
                out = F.relu(out, inplace=True)
                out = F.conv2d(out, weights['layer3.%d.conv2.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                 weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i],training=True)    
                if i==0:
                    residual = F.conv2d(x, weights['layer3.%d.downsample.0.weight'%i], bias=None, stride=2)  
                    residual = F.batch_norm(residual, self.layer3[i].downsample[1].running_mean, self.layer3[i].downsample[1].running_var, 
                                 weights['layer3.%d.downsample.1.weight'%i], weights['layer3.%d.downsample.1.bias'%i],training=True)  
                x = out + residual  
                x = F.relu(x, inplace=True)
            feat3 = x
                
            #layer 4
            for i in range(2):
                residual = x
                if i == 0:
                    out = F.conv2d(x, weights['layer4.%d.conv1.weight'%i], bias=None, stride=2, padding=1)
                else:
                    out = F.conv2d(x, weights['layer4.%d.conv1.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer4[i].bn1.running_mean, self.layer4[i].bn1.running_var, 
                                 weights['layer4.%d.bn1.weight'%i], weights['layer4.%d.bn1.bias'%i],training=True)     
                out = F.relu(out, inplace=True)
                out = F.conv2d(out, weights['layer4.%d.conv2.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer4[i].bn2.running_mean, self.layer4[i].bn2.running_var, 
                                 weights['layer4.%d.bn2.weight'%i], weights['layer4.%d.bn2.bias'%i],training=True)    
                if i==0:
                    residual = F.conv2d(x, weights['layer4.%d.downsample.0.weight'%i], bias=None, stride=2)  
                    residual = F.batch_norm(residual, self.layer4[i].downsample[1].running_mean, self.layer4[i].downsample[1].running_var, 
                                 weights['layer4.%d.downsample.1.weight'%i], weights['layer4.%d.downsample.1.bias'%i],training=True)  
                x = out + residual  
                x = F.relu(x, inplace=True)
            feat4 = x
                
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            feat = x.view(x.size(0), -1)
            x = F.linear(feat, weights['fc.weight'], weights['fc.bias'])    
            
            if SD == None:
                return x
            else:
                atten = self.attention(feat2)
                feat2 = atten*feat2
                SD_x = self.scala(feat2)
                SD_feat = torch.flatten(SD_x, 1)
                SD_x = self.fc2(SD_feat)
                
                return x, SD_x        
            
class ResNetCifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCifar, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)


        feat = torch.flatten(x, 1)
        x = self.fc(feat)


        return x


                
    

def ResNet18_cifar(class_num, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar(BasicBlock, [2, 2, 2, 2], num_classes = class_num, **kwargs)



def ResNet18_SD_cifar(class_num, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar_SD(BasicBlock, [2, 2, 2, 2], num_classes = class_num, **kwargs)