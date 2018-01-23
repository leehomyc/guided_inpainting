### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


# noinspection PyAttributeOutsideInit,PyPep8Naming
class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none':  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        if opt.label_nc != 0:
            input_nc = opt.label_nc
        elif opt.model == 'inpainting_grid':
            input_nc = 24
        else:
            input_nc = 3
        # input_nc = opt.label_nc if opt.label_nc != 0 else 3

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, dilation=opt.dilation,
                                      interpolated_conv=opt.interpolated_conv)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

                # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            if self.opt.use_local_discriminator:
                self.loss_names = ['G_GAN', 'G_GAN_local', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_real_local', 'D_fake', 'D_fake_local']  #noqa
            else:
                self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print(
                    '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params': [value], 'lr': opt.lr}]
                    else:
                        params += [{'params': [value], 'lr': 0.0}]
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D                        
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)

        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        fake_image = self.netG.forward(input_concat)

        # If we want to crop the hole and paste it back to the original image.
        hole_y_begin = int(self.opt.fineSize / 4 + self.opt.overlapPred)
        hole_y_end = int(self.opt.fineSize / 2 + self.opt.fineSize / 4 - self.opt.overlapPred)
        hole_x_begin = hole_y_begin
        hole_x_end = hole_y_end
        hole_height = hole_y_end - hole_y_begin
        hole_width = hole_x_end - hole_x_begin

        # Local discriminator takes in a patch that is slightly larger than the hole
        local_discriminator_y_begin = int(hole_y_begin - hole_height / 4)
        local_discriminator_x_begin = int(hole_x_begin - hole_width / 4)
        local_discriminator_y_end = int(hole_y_end + hole_height / 4)
        local_discriminator_x_end = int(hole_x_end + hole_width / 4)
        if self.opt.keep_hole_only is True:
            # Create a mask with zeros in the hole region and ones in the boundary
            mask = np.ones((self.opt.batchSize, 3, self.opt.fineSize, self.opt.fineSize))
            mask[:, :, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 0
            mask = torch.from_numpy(mask).float().cuda()
            mask = Variable(mask)
            # add the image and mask together.
            fake_image = input_label * mask + fake_image * (1 - mask)

        if self.opt.use_local_discriminator:
            real_hole = input_label[:, :, local_discriminator_y_begin: local_discriminator_y_end, local_discriminator_x_begin: local_discriminator_x_end]  # noqa
            fake_hole = fake_image[:, :, local_discriminator_y_begin: local_discriminator_y_end, local_discriminator_x_begin: local_discriminator_x_end]  # noqa

        # Fake Detection and Loss. This loss is to update the discriminator.
        pred_fake_pool = self.discriminate(input_label, real_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        if self.opt.use_local_discriminator:
            pred_fake_local = self.discriminate(real_hole, fake_hole)
            loss_D_fake_local = self.criterionGAN(pred_fake_local, False)

        # Real Detection and Loss. This loss is to update the discriminator.
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        if self.opt.use_local_discriminator:
            pred_real_local = self.discriminate(real_hole, fake_hole)
            loss_D_real_local = self.criterionGAN(pred_real_local, True)

        # GAN loss (Fake Passability Loss). This loss is to update the generator.
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        if self.opt.use_local_discriminator:
            pred_fake_local = self.netD.forward(torch.cat((real_hole, fake_hole), dim=1))
            loss_G_GAN_local = self.criterionGAN(pred_fake_local, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW The names of the losses are ['G_GAN', 'G_GAN_local',
        # 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_real_local', 'D_fake', 'D_fake_local']
        if self.opt.use_local_discriminator:
            return [[loss_G_GAN, loss_G_GAN_local, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_real_local, loss_D_fake, loss_D_fake_local], None if not infer else fake_image]  # noqa
        else:
            return [[loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake], None if not infer else fake_image]

    def inference(self, label, inst):
        # Encode Inputs        
        input_label, inst_map, _, _ = self.encode_input(Variable(label), Variable(inst), infer=True)

        # Fake Generation
        if self.use_features:
            # sample clusters from precomputed features             
            feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        fake_image = self.netG.forward(input_concat)

        hole_y_begin = int(self.opt.fineSize / 4 + self.opt.overlapPred)
        hole_y_end = int(self.opt.fineSize / 2 + self.opt.fineSize / 4 - self.opt.overlapPred)
        hole_x_begin = hole_y_begin
        hole_x_end = hole_y_end

        if self.opt.keep_hole_only is True:
            # Create a mask with zeros in the hole region and ones in the boundary
            mask = np.ones((self.opt.batchSize, 3, self.opt.fineSize, self.opt.fineSize))
            mask[:, :, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 0
            mask = torch.from_numpy(mask).float().cuda()
            mask = Variable(mask)
            # add the image and mask together.
            fake_image = input_label * mask + fake_image * (1 - mask)
        return fake_image

    def sample_features(self, inst):
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = torch.cuda.FloatTensor(1, self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == i).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == i).nonzero()
            num = idx.size()[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
