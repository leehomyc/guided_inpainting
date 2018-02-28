import torch
from torch.autograd import Variable

from . import networks
from .base_model import BaseModel
from inpainting.util.image_pool import ImagePool


# noinspection PyAttributeOutsideInit,PyPep8Naming
class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none':
            # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        self.use_mask = opt.use_mask
        if opt.label_nc != 0:
            input_nc = opt.label_nc
        else:
            input_nc = 3

        # define networks

        #####################
        # Generator network #
        #####################
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        if self.use_mask:
            netG_input_nc += 1
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf,
                                      opt.netG,
                                      opt.n_downsample_global,
                                      opt.n_blocks_global,
                                      opt.norm,
                                      gpu_ids=self.gpu_ids,
                                      dilation=opt.dilation,
                                      interpolated_conv=opt.interpolated_conv)

        #########################
        # Discriminator network #
        #########################
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf,
                                          opt.n_layers_D,
                                          opt.global_layer, opt.norm,
                                          use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss,
                                          opt.globalGAN_loss,
                                          gpu_ids=self.gpu_ids)

        print('---------- Networks initialized -------------')

        ################
        # Load Network #
        ################
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch,
                                  pretrained_path)

        ######## set loss functions and optimizers #############
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError(
                    "Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            ######### set loss functions ###########

            #################
            # PatchGAN Loss #
            #################
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan,
                                                 tensor=self.Tensor)
            #########################
            # Feature matching Loss #
            #########################
            self.criterionFeat = torch.nn.L1Loss()

            ############
            # VGG Loss #
            ############
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss()

            #############################
            # Image Reconstruction loss.#
            #############################
            if opt.recon_loss:
                self.criterionRecon = torch.nn.L1Loss()

            self.loss_names = \
                ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_recon', 'D_real', 'D_fake']

            ######### set optimizers ###########
            ##############
            # Optimize G #
            ##############
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

            ##############
            # Optimize D #
            ##############
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

    def encode_input(self, input_image, input_mask=None, original_image=None,
                     infer=False):

        input_label = input_image.data.cuda()

        # get edges from instance map
        if not self.opt.isTrain:
            input_mask = Variable(input_mask.data.cuda())

        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if original_image is not None:
            original_image = Variable(original_image.data.cuda())

        return input_label, input_mask, original_image

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, input_image, input_mask, original_image, infer=False):
        # Encode Inputs
        input_image_e, input_mask_e, original_image_e = \
            self.encode_input(input_image, input_mask, original_image)

        # Fake Generation
        input_concat = input_image_e

        fake_image = self.netG.forward(input_concat)

        if self.opt.model == 'inpainting_object' or 'inpainting_guided':
            fake_image = original_image_e * (1 - input_mask_e) + \
                         fake_image * input_mask_e

        ###############################
        # PatchGAN Discriminator loss.#
        ###############################
        pred_fake_pool = self.discriminate(input_image_e, fake_image,
                                           use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        pred_real = self.discriminate(input_image_e, original_image_e)
        loss_D_real = self.criterionGAN(pred_real, True)

        ###########################
        # PatchGAN Generator loss.#
        ###########################
        pred_fake = self.netD.forward(
            torch.cat((input_image_e, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        #############################
        # GAN feature matching loss.#
        #############################
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += \
                        D_weights * \
                        feat_weights * \
                        self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach()) * \
                        self.opt.lambda_feat

        #############################
        # VGG feature matching loss.#
        #############################
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = \
                self.criterionVGG(fake_image, original_image_e) * \
                self.opt.lambda_feat

        #############################
        # Image Reconstruction loss.#
        #############################
        loss_G_recon = 0
        if self.opt.recon_loss:
            loss_G_recon += \
                self.criterionRecon(fake_image, original_image_e) * \
                self.opt.lambda_recon

        return [[loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_recon,
                 loss_D_real, loss_D_fake], None if not infer else fake_image]

    def inference(self, input_image, input_mask):
        # Encode Inputs
        input_image_e, input_mask_e, _, = self.encode_input(
            Variable(input_image), Variable(input_mask), infer=True)

        input_concat = input_image_e
        fake_image = self.netG.forward(input_concat)
        if self.opt.model == 'inpainting_object' or 'inpainting_guided':
            fake_image = \
                input_image_e * (1 - input_mask_e) + fake_image * input_mask_e

        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations,
        # also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr,
                                            betas=(self.opt.beta1, 0.999))
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
