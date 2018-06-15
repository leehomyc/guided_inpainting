"""Training files for inpainting."""

from collections import OrderedDict
import os
import numpy as np
import time
import torch
from torch.autograd import Variable

from inpainting.data.data_loader import CreateDataLoader
from inpainting.models.models import create_model
from inpainting.options.train_options import TrainOptions
import inpainting.util.util as util
from inpainting.util.visualizer import Visualizer

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    # noinspection PyBroadException
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',',
                                             dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0

        ############## Forward Pass ######################
        if opt.use_seg:
            losses, generated = model(Variable(data['input']),
                                      Variable(data['mask']),
                                      Variable(data['image']),
                                      Variable(data['input_seg']),
                                      infer=save_fake)
        elif opt.use_conditional_image:
            losses, generated = model(Variable(data['input']),
                                      Variable(data['mask']),
                                      Variable(data['image']),
                                      input_conditional_image=Variable(data['masked_image']),
                                      infer=save_fake)

        elif opt.use_seg_and_conditional_image:
            losses, generated = model(Variable(data['input']),
                                      Variable(data['mask']),
                                      Variable(data['image']),
                                      input_seg=Variable(data['input_seg']),
                                      input_conditional_image=Variable(data['conditional_image']),
                                      infer=save_fake)


        else:
            losses, generated = model(Variable(data['input']),
                                  Variable(data['mask']),
                                  Variable(data['image']),
                                  infer=save_fake)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in
                  losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict[
            'G_VGG'] + loss_dict['G_recon'] + loss_dict['Perceptual']

        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == 0:
            errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in
                      loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            if opt.use_pretrained_model:
                visuals_list = [('input_image_original',
                                        util.tensor2label(data['input_original'][0],
                                                          opt.label_nc)),
                                       ('input_image',
                                        util.tensor2label(data['input'][0],
                                                          opt.label_nc)),
                                       ('synthesized_image',
                                        util.tensor2label(generated.data[0],
                                                          opt.label_nc)),
                                       ('real_image',
                                        util.tensor2label(data['image'][0],
                                                          opt.label_nc))]
                                       # ('synthesized_image',
                                       #  util.tensor2im(generated.data[0])),
                                       # ('real_image',
                                       #  util.tensor2im(data['image'][0]))]
                if opt.use_seg:
                    visuals_list.append(('input_segmentation',
                                        util.tensor2label(data['input_seg'][0],
                                                          opt.seg_nc)))
                if opt.use_conditional_image:
                    visuals_list.append(('conditional_image',
                                        util.tensor2im(data['masked_image'][0])))   
                if opt.use_seg_and_conditional_image:
                    visuals_list.append(('input_segmentation',
                                        util.tensor2label(data['input_seg'][0],
                                                          opt.seg_nc)))
                    visuals_list.append(('conditional_image',
                                        util.tensor2im(data['conditional_image'][0]))) 


                visuals = OrderedDict(visuals_list)
            else:
                visuals_list = [('input_image',
                                    util.tensor2label(data['input'][0],
                                                      opt.label_nc)),
                                       ('synthesized_image',
                                        util.tensor2label(generated.data[0],
                                                          opt.label_nc)),
                                       ('real_image',
                                        util.tensor2label(data['image'][0],
                                                          opt.label_nc))]
                                   # ('synthesized_image',
                                   #  util.tensor2im(generated.data[0])),
                                   # ('real_image',
                                   #  util.tensor2im(data['image'][0]))]
                if opt.use_seg:
                    visuals_list.append(('input_segmentation',
                                    util.tensor2label(data['input_seg'][0],
                                                          opt.seg_nc)))
                if opt.use_conditional_image:
                    visuals_list.append(('conditional_image',
                                        util.tensor2im(data['masked_image'][0]))) 
                if opt.use_seg_and_conditional_image:
                    visuals_list.append(('input_segmentation',
                                        util.tensor2label(data['input_seg'][0],
                                                          opt.seg_nc)))
                    visuals_list.append(('conditional_image',
                                        util.tensor2im(data['conditional_image'][0])))  

                visuals = OrderedDict(visuals_list)
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (
                epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (
            epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    # ## instead of only training the local enhancer, train the entire
    # network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
