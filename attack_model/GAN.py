import torch.nn as nn
import torch.nn.functional as F


#MARK: AdvGAN / PA-GAN
class AdvDiscriminator(nn.Module):
    def __init__(self, image_nc):
        super(AdvDiscriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvGenerator(nn.Module):
    def __init__(self, gen_input_nc, image_nc, img_size):
        super(AdvGenerator, self).__init__()
        self.gen_input_nc = gen_input_nc
        self.img_size = img_size
        self.encoder = nn.Sequential(
            nn.Conv2d(gen_input_nc, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )

        self.bottle_neck = nn.Sequential(
            ResnetBlock(128),
            ResnetBlock(128),
            ResnetBlock(128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, image_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # perturbation: -1 ~ 1
        )

        if self.output_shape_mismatch():
            self.upsample_to_target = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)
        else:
            self.upsample_to_target = nn.Identity()

    def output_shape_mismatch(self):
        with torch.no_grad():
            dummy = torch.randn(1, self.gen_input_nc, self.img_size, self.img_size)
            x = self.encoder(dummy)
            x = self.bottle_neck(x)
            x = self.decoder(x)
            return x.shape[-1] != self.img_size or x.shape[-2] != self.img_size

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        x = self.upsample_to_target(x)
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out