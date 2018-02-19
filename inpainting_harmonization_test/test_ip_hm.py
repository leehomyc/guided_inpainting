"""This is the file to test inpainting and harmonization jointly. The input
is an image pair, where one image is the foreground object, which we have to
specify the annotation ID of the object. The second image is the background,
which we have to specify the new location of the upper left corner and the
height and width to inpaint the object. Please refer to
inpainting_dataset_test.py for reference. """

from collections import OrderedDict
import os

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.test_options import TestOptions
from util import html
import util.util as util
from util.visualizer import Visualizer

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase,
                                                             opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    print('processing image {}'.format(i))
    generated = model.inference(data['input'], data['mask'])
    visuals = OrderedDict([('input', util.tensor2label(data['input'][0], opt.label_nc)),  # noqa 501
                           ('mask', util.tensor2im(data['mask'][0], normalize=False)),  # noqa 501
                           ('synthesized_image', util.tensor2im(generated.data[0]))])   # noqa 501
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)

print('saving')
webpage.save()
