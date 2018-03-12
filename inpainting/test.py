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
    generated = model.inference(data['input'], data['mask'])
    visuals = OrderedDict([('input_image', util.tensor2label(data['input'][0], opt.label_nc)),  # noqa 501
    					   ('image', util.tensor2label(data['image'][0], opt.label_nc)),
                           ('input_mask', util.tensor2label(data['mask'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])  # noqa 501
    img_path = data['path']
    print('process image... {}:{}'.format(i, img_path))
    visualizer.save_images(webpage, visuals, img_path)

print('saving')
webpage.save()
