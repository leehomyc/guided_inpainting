"""This is the file to test inpainting and harmonization jointly. The input
is an image pair, where one image is the foreground object, which we have to
specify the annotation ID of the object. The second image is the background,
which we have to specify the new location of the upper left corner and the
height and width to inpaint the object. Please refer to
inpainting_dataset_test.py for reference. """

from collections import OrderedDict
import os

from inpainting_harmonization_test.data.data_loader import CreateDataLoader
from inpainting.models.models import create_model as create_inpainting_model
from harmonization.models.models import \
    create_model as create_harmonization_model
from inpainting_harmonization_test.options.test_options import TestOptions
from inpainting_harmonization_test.util import html
import inpainting_harmonization_test.util.util as util
from inpainting_harmonization_test.util.visualizer import Visualizer

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

opt.model = 'inpainting_guided'
opt.checkpoints_dir = opt.ip_checkpoints_dir
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model_inpainting = create_inpainting_model(opt)

opt.model = 'inpainting_harm'
opt.checkpoints_dir = opt.harm_checkpoints_dir
opt.interpolated_conv = False
opt.dilation = 1
model_harmonization = create_harmonization_model(opt)

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
    generated = model_inpainting.inference(data['input'], data['mask'])
    visuals = OrderedDict(
        [('input', util.tensor2label(data['input'][0], opt.label_nc)),
         # noqa 501
         ('mask_object', util.tensor2im(data['mask_composite_object'][0], normalize=False)),  # noqa 501
         ('synthesized_image', util.tensor2im(generated.data[0]))])  # noqa 501
    img_path = data['path']

    inpainted_image = generated.data[0]
    inpainted_image = inpainted_image[None, :, :, :]

    generated = model_harmonization.inference(inpainted_image, data['mask_composite_object'])  # noqa 501

    visuals['synthesized_image_final'] = util.tensor2im(generated.data[0])
    visualizer.save_images(webpage, visuals, img_path)

print('saving')
webpage.save()
