from collections import OrderedDict
import os

from pix2pixHD.data.data_loader import CreateDataLoader
from pix2pixHD.options.test_options import TestOptions
from pix2pixHD.models.models import create_model
from pix2pixHD.util import html
import pix2pixHD.util.util as util
from pix2pixHD.util.visualizer import Visualizer

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
                           ('synthesized_image', util.tensor2im(generated.data[0]))])  # noqa 501
    img_path = data['path']
    print('process image... {}:{}'.format(i, img_path))
    visualizer.save_images(webpage, visuals, img_path)

print('saving')
webpage.save()
