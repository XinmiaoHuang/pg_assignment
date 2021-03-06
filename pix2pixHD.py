import torch
import functools
import torch.nn as nn
from utils import ResBlock, StyleBlock, Dense, SPADE_Resblock
from utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_blocks=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()
        sequence = []
        sequence += nn.Sequential(*[nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True)])
        n_mult = 1
        for i in range(n_blocks):
            sequence += nn.Sequential(*[nn.Conv2d(ndf * n_mult, ndf * n_mult * 2, 4, 2, 1, bias=False),
                         norm_layer(ndf * n_mult * 2),
                         nn.LeakyReLU(0.2, inplace=True)])
            n_mult *= 2
            # sequence += [ResBlock(ndf * n_mult)]
        sequence += [nn.Conv2d(ndf * n_mult, 1, 4, 1, 1, bias=False)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.ModuleList(sequence)

    def forward(self, input):
        x = input
        for i in range(len(self.model)):
            x = self.model[i](x)
        return x

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_dnet=3, n_blocks=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(MultiScaleDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ndf = ndf
        self.n_dnet = n_dnet
        self.n_block = n_blocks
        self.norm_layer = norm_layer
        self.use_sigmoid = use_sigmoid

        # discriminator1
        sequence1 = []
        sequence1 += nn.Sequential(*[nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True)])
        n_mult = 1
        for _ in range(n_blocks):
            sequence1 += nn.Sequential(*[nn.Conv2d(ndf * n_mult, ndf * n_mult * 2, 4, 2, 1, bias=False),
                        norm_layer(ndf * n_mult * 2),
                        nn.LeakyReLU(0.2, inplace=True)])
            n_mult *= 2
        sequence1 += [nn.Conv2d(ndf * n_mult, 1, 4, 1, 1, bias=False)]
        if use_sigmoid:
            sequence1 += [nn.Sigmoid()]
        self.d1 = nn.ModuleList(sequence1)

        # discriminator2
        sequence2 = []
        sequence2 += nn.Sequential(*[nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True)])
        n_mult = 1
        for _ in range(n_blocks):
            sequence2 += nn.Sequential(*[nn.Conv2d(ndf * n_mult, ndf * n_mult * 2, 4, 2, 1, bias=False),
                        norm_layer(ndf * n_mult * 2),
                        nn.LeakyReLU(0.2, inplace=True)])
            n_mult *= 2
        sequence2 += [nn.Conv2d(ndf * n_mult, 1, 4, 1, 1, bias=False)]
        if use_sigmoid:
            sequence2 += [nn.Sigmoid()]
        self.d2 = nn.ModuleList(sequence2)

        # discriminator3
        sequence3 = []
        sequence3 += nn.Sequential(*[nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True)])
        n_mult = 1
        for _ in range(n_blocks):
            sequence3 += nn.Sequential(*[nn.Conv2d(ndf * n_mult, ndf * n_mult * 2, 4, 2, 1, bias=False),
                        norm_layer(ndf * n_mult * 2),
                        nn.LeakyReLU(0.2, inplace=True)])
            n_mult *= 2
        sequence3 += [nn.Conv2d(ndf * n_mult, 1, 4, 1, 1, bias=False)]
        if use_sigmoid:
            sequence3 += [nn.Sigmoid()]
        self.d3 = nn.ModuleList(sequence3)

        self.d_net = [self.d1, self.d2, self.d3]
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        interFeat = []
        res = []
        for i in range(self.n_dnet):
            x = input
            output = []
            dnet = self.d_net[i]
            for j in range(len(dnet)):
                x = dnet[j](x)
                if j in [1, 4, 7, 10, 11]:         # feature for fm loss
                    output.append(x)
            interFeat.append(output)
            res.append(x)
            if i is not self.n_dnet-1:
                input = self.downsample(input)
        return interFeat, res

class Discriminator_t(nn.Module):
    def __init__(self, input_nc, ndf=64, n_blocks=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator_t, self).__init__()
        sequence = []
        sequence += nn.Sequential(*[spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False)),
                                    nn.LeakyReLU(0.2, inplace=True)])
        n_mult = 1
        for _ in range(n_blocks):
            sequence += nn.Sequential(*[
                         spectral_norm(nn.Conv2d(ndf * n_mult, ndf * n_mult * 2, 4, 2, 1, bias=False)),
                         norm_layer(ndf * n_mult * 2),
                         nn.LeakyReLU(0.2, inplace=True)])
            n_mult *= 2
            # sequence += [ResBlock(ndf * n_mult)]
        sequence += [nn.Conv2d(ndf * n_mult, 1, 4, 1, 1, bias=False)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.ModuleList(sequence)

    def forward(self, input):
        x = input
        for i in range(len(self.model)):
            x = self.model[i](x)
        return x

class Discriminator_p(nn.Module):
    def __init__(self, input_nc, ndf=64, n_blocks=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator_p, self).__init__()
        sequence = []
        sequence += nn.Sequential(*[spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False)),
                                    nn.LeakyReLU(0.2, inplace=True)])
        n_mult = 1
        for _ in range(n_blocks):
            sequence += nn.Sequential(*[
                         spectral_norm(nn.Conv2d(ndf * n_mult, ndf * n_mult * 2, 4, 2, 1, bias=False)),
                         norm_layer(ndf * n_mult * 2),
                         nn.LeakyReLU(0.2, inplace=True)])
            n_mult *= 2
            # sequence += [ResBlock(ndf * n_mult)]
        sequence += [nn.Conv2d(ndf * n_mult, 1, 4, 1, 1, bias=False)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.ModuleList(sequence)

    def forward(self, input):
        x = input
        for i in range(len(self.model)):
            x = self.model[i](x)
        return x


class pix2pix(nn.Module):
    def __init__(self, input_nc, n_layer=3, norm_layer=nn.BatchNorm2d, ngf=64):
        super(pix2pix, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf

        encoder = [nn.Conv2d(input_nc, ngf, 3, 2, 1, bias=False),
                   nn.ReLU(True)
                   ]
        n_mult = 1
        for _ in range(n_layer):
            encoder += [nn.Conv2d(ngf * n_mult, ngf * (n_mult * 2), 3, 2, 1, bias=False),
                        norm_layer(ngf * (n_mult * 2)),
                        nn.ReLU(True)]
            n_mult = n_mult * 2
        self.encoder = nn.Sequential(*encoder)

        resnet = []
        n_blocks = 6
        for _ in range(n_blocks):
            resnet += [ResBlock(n_mult * ngf)]
        self.resnet = nn.Sequential(*resnet)

        decoder = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                   nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
                   norm_layer(ngf * 8),
                   nn.ReLU(True)]
        n_mult = 8
        for _ in range(n_layer):
            decoder += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(ngf * n_mult, ngf * (n_mult // 2), 3, 1, 1, bias=False),
                        norm_layer(ngf * (n_mult // 2)),
                        nn.ReLU(True)]
            n_mult = n_mult // 2

        decoder += [nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
                    nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, img, pose, style):
        x = torch.cat((img, pose), dim=1)
        x = self.encoder(x)
        x = self.resnet(x)
        out = self.decoder(x)
        return out


class pix2pixStyle(nn.Module):
    def __init__(self, input_nc, n_layer=3, norm_layer=nn.BatchNorm2d, ngf=64):
        super(pix2pixStyle, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf

        encoder = [nn.ReflectionPad2d(3),
                   spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0)),
                   nn.ReLU(True),
                   spectral_norm(nn.Conv2d(ngf, ngf, 3, 2, 1, bias=False)),
                   norm_layer(ngf),
                   nn.ReLU(True),
                   ]
        n_mult = 1
        for _ in range(n_layer):
            ngf_in = min(ngf * n_mult, ngf * 4)
            ngf_out = min(ngf * (n_mult * 2), ngf * 4)
            encoder += [spectral_norm(nn.Conv2d(ngf_in, ngf_out, 3, 2, 1, bias=False)),
                        norm_layer(ngf_out),
                        nn.ReLU(True)]
            n_mult = n_mult * 2
        self.encoder = nn.Sequential(*encoder)


        style_encoder = [spectral_norm(nn.Conv2d(21, ngf, 3, 1, 1, bias=False)),
                         norm_layer(ngf),
                         nn.ReLU(True)
                        ]
        n_mult = 1
        for _ in range(n_layer):
            ngf_in = min(ngf * n_mult, ngf * 4)
            ngf_out = min(ngf * (n_mult * 2), ngf * 4)
            style_encoder += [Dense(ngf_in, ngf_in),
                              spectral_norm(nn.Conv2d(ngf_in, ngf_out, 3, 2, 1, bias=False)),
                              norm_layer(ngf_out),
                              nn.ReLU(True)]
            n_mult = n_mult * 2
        self.style_encoder = nn.Sequential(*style_encoder)

        resnet = []
        n_blocks = 6
        for _ in range(n_blocks):
            resnet += [StyleBlock(4 * ngf)]
        self.resnet = nn.ModuleList(resnet)

        decoder = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                   spectral_norm(nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1, bias=False)),
                   norm_layer(ngf * 4),
                   nn.ReLU(True)]

        decoder += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    spectral_norm(nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1, bias=False)),
                    norm_layer(ngf * 4),
                    nn.ReLU(True)]
        n_mult = 4
        for _ in range(n_layer-1):
            decoder += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        spectral_norm(nn.Conv2d(ngf * n_mult, ngf * (n_mult // 2), 3, 1, 1, bias=False)),
                        norm_layer(ngf * (n_mult // 2)),
                        nn.ReLU(True)]
            n_mult = n_mult // 2

        decoder += [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0),
                    nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, img, pose, style):
        sty_in = torch.cat((img, pose), dim=1)
        sty = self.style_encoder(sty_in)
        x = self.encoder(pose)
        for i in range(len(self.resnet)):
            x = self.resnet[i](x, sty)
        out = self.decoder(x)
        return out