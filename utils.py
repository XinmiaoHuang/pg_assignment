import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


label_colors = [(0,0,0), (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0),
                (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128),
                (0,128,0), (0,0,255), (51,170,221), (0,255,255), (85,255,170),
                (170,255,85), (255,255,0), (255,170,0)]

labels = []
for colors in label_colors:
    labels.append((colors[0]*1+colors[1]*7+colors[2]*9)/17/255.)
labels = np.array(labels).astype(np.float16)

def normalize(x):
    return 2.0 * x - 1

def denormalize(x):
    return (x + 1.0) / 2.0

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, norm_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW, pytorch is channel first language
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels, k_size=3, stride=1, padding=1,
                 dilation=1, norm_layer=nn.BatchNorm2d, add_noise=False):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channels, channels, k_size, stride, padding, dilation),
            norm_layer(channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, k_size, stride, padding, dilation),
            norm_layer(channels),
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.add_noise = add_noise

    def addNoise(self, mu, rate=0.1):
        eps = torch.randn_like(mu)
        return mu + eps * rate

    def forward(self, x):
        residual = x
        if self.add_noise:
            x = self.addNoise(x)
        x = self.conv_block1(x)
        out = self.conv_block2(x)
        out = self.relu(out + residual)
        return out

class StyleBlock(nn.Module):
    def __init__(self, channels, k_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channels, channels, k_size, stride, padding, dilation),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, k_size, stride, padding, dilation),
        )
        self.l_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, style_mean, style_std):
        residual = x
        x = self.conv_block1(x)
        x = adaptive_instance_normalization(x, style_mean, style_std)
        x = self.l_relu(x)
        x = self.conv_block2(x)
        x = adaptive_instance_normalization(x, style_mean, style_std)
        out = self.l_relu(x + residual)
        return out

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.LeakyReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(growth_rate)),
        self.add_module('relu2', nn.LeakyReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
 
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class Dense(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shape = x.size()
        b, c, h, w = shape[0], shape[1], shape[2], shape[3]
        x = x.permute((0, 2, 3, 1))
        # x = x.view(b*h*w, -1)
        linear = nn.Linear(c, self.out_channels).cuda()
        out = linear(x)
        out = out.permute((0, 3, 1, 2))
        out = self.bn(out)
        return out


def make_parts_shape(parts, poses=None):
    # 最后一维是真个人的shape，不要
    if len(parts) == 21: parts = parts[:-1 , : , :]
    shape_labels = np.argmax(parts, axis=0)
    shape = np.zeros(parts[0].shape + (3,))
    for label in range(len(label_colors)):
        shape[shape_labels==label] = label_colors[label]
    if poses is not None:
        for k in range(3):
            if k == 0: continue                         # 不要body红色通道
            shape[poses[k]>0] = 255.
    return np.array(shape/255.).transpose((2,0,1))


def make_shape_parts(img, test_output=False,
                    image_data_format='channels_first'):
    # 单张shape图拆分parts(21个)
    if image_data_format == 'channels_last':
        img = np.transpose(img, (2, 0, 1))
    if img.dtype == np.uint8:
        img = img / 255.
        img = img.astype(np.float16)
    img = (img[0,:,:]*1+img[1,:,:]*7+img[2,:,:]*9)/17
    # img.shape = (256, 256), from [0, 1]
    if test_output:
        cv2.imshow('GrayImage', img)                    # Debug测试输出

    # shape_parts -> Zero Matrix (16, 256, 256)
    size = (len(label_colors), img.shape[0], img.shape[1])
    shape_parts = np.zeros(size, dtype=np.float16)
    for i, label in enumerate(labels):
        shape_parts[i][img==label] = 1.0
    # 将最后一个设置为人身
    # shape_parts[-1][img!=labels[0]] = 1.0
    return shape_parts


"""
reference:
    https://github.com/naoto0804/pytorch-AdaIN/
"""
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_mean, style_std):
    # assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

