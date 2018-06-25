from collections import OrderedDict
import os
import numpy as np

from inpainting.data.data_loader import CreateDataLoader
from inpainting.options.test_options import TestOptions
from inpainting.models.models import create_model
from inpainting.util import html
import inpainting.util.util as util
from inpainting.util.visualizer import Visualizer

np.random.seed(0)

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))  # noqa 501
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))  # noqa 501

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.use_seg:
        generated = model.inference(data['input'], data['mask'], input_seg=data['input_seg'])
    elif opt.use_conditional_image:
        generated = model.inference(data['input'], data['mask'], input_conditional_image=data['masked_image'])
    elif opt.use_seg_and_conditional_image:
        generated = model.inference(data['input'], data['mask'], input_seg=data['input_seg'], input_conditional_image=data['masked_image'])
    else:
        generated = model.inference(data['input'], data['mask'])
    visuals_list = [('input_image', util.tensor2label(data['input'][0], opt.label_nc)),  # noqa 501
                 ('image', util.tensor2label(data['image'][0], opt.label_nc)),
                           # ('input_mask', util.tensor2label(data['mask'][0], opt.label_nc)),
                           ('input_mask', util.tensor2im(data['mask'][0], normalize=False)),
                           # ('synthesized_image', util.tensor2im(generated.data[0]))]
                           ('synthesized_image',
                                        util.tensor2label(generated.data[0],
                                                          0 if opt.output_nc==3 else opt.output_nc))]
    # if opt.model == 'inpainting_cityscapes_predict_segmentation':
    if 'predict_segmentation' in opt.model:
      visuals_list.append(('original_image',
                                    util.tensor2im(data['original_image'][0])))
      visuals_list.append(('predicted_label',
                                    util.tensor2segLabel(generated.data[0])))
                                                         
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
    visuals = OrderedDict(visuals_list)  # noqa 501
    img_path = data['path']
    print('process image... {}:{}'.format(i, img_path))
    visualizer.save_images(webpage, visuals, img_path)

print('saving')
webpage.save()
