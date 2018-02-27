import functools
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models


#############
# Functions #
#############
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)  # noqa 501
    return norm_layer


#############
# Generator #
#############
def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3,
             n_blocks_global=9, norm='instance', gpu_ids=[], dilation=1,
             interpolated_conv=False):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global,
                               n_blocks_global, norm_layer,
                               dilation=dilation,
                               interpolated_conv=interpolated_conv)
    else:
        raise Exception('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


#################
# Discriminator #
#################

# noinspection PyPep8Naming
def define_D(input_nc, ndf, n_layers_D, global_layer=6, norm='instance', use_sigmoid=False,
             num_D=1, getIntermFeat=False, globalGAN_loss=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, global_layer, norm_layer,
                                   use_sigmoid, num_D, getIntermFeat, globalGAN_loss)

    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


###################
# PatchGAN Losses #
###################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += \
                self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


##############################################################################
# Generator
##############################################################################
# noinspection PyShadowingBuiltins
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3,
                 n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', dilation=1, interpolated_conv=False):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7,
                                                  padding=0), norm_layer(ngf),
                 activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if interpolated_conv is True:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=1, padding=1),
                          nn.MaxPool2d(2),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          ]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1),
                          norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  activation=activation, norm_layer=norm_layer,
                                  dilation=dilation)]

        ### upsample
        # Harry: use interpolated convolution instead of transposed convolution
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if interpolated_conv is True:
                model += [
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
            else:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2, padding=1,
                                             output_padding=1),
                          norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


#########################
# Define a resnet block #
#########################
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True),
                 use_dropout=False, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                activation, use_dropout,
                                                dilation)

    def build_conv_block(self, dim, padding_type, norm_layer, activation,
                         use_dropout, dilation):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1 + int(dilation / 2))]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1 + int(dilation / 2))]
        elif padding_type == 'zero':
            p = 1 + int(dilation / 2)
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=dilation),
            norm_layer(dim),
            activation]
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
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


##########################
# Define a discriminator #
##########################
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, global_layer=6, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False, globalGAN_loss=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.globalGAN_loss = globalGAN_loss

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer,
                                       use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j),
                            getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        if globalGAN_loss:
            netD_global = GlobalDiscriminator(input_nc, ndf, global_layer, norm_layer,
                                              use_sigmoid)
            setattr(self,'global', netD_global.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1],
                                       count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input

        if self.globalGAN_loss:
            model = getattr(self, 'global')
            result.append(self.singleD_forward(model, input_downsampled))

        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self,
                                 'scale' + str(num_D - 1 - i) + '_layer' + str(
                                     j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
             nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

# Defines the PatchGAN discriminator with the specified arguments.
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class GlobalDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, global_layer=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, getIntermFeat=False):
        super(GlobalDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.global_layer = global_layer

        kw = 5
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
             nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, global_layer):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        # output: 4*4*512

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
