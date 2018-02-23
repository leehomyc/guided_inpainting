"""This is the file to test inpainting and harmonization jointly. The input is an image pair, where one image is the
foreground object, which we have to specify the annotation ID of the object. The second image is the background,
which we have to specify the new location of the upper left corner and the height and width to inpaint the object.
Please refer to inpainting_dataset_test.py for reference. """

from collections import OrderedDict
import os

import caffe
import numpy as np
import scipy
import scipy.misc

from pix2pixHD.data.data_loader import CreateDataLoader
from pix2pixHD.models.models import create_model
from pix2pixHD.options.test_options import TestOptions
from pix2pixHD.util import html
import pix2pixHD.util.util as util
from pix2pixHD.util.visualizer import Visualizer

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
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('DeepHarmonization/model/deploy_512.prototxt',
                'DeepHarmonization/model/harmonize_iter_200000.caffemodel', caffe.TEST)  # noqa

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    generated = model.inference(data['label'], data['inst'])
    visuals = OrderedDict([('input_label', util.tensor2label(data['image_composite_with_background'][0], opt.label_nc)),
                           ('input_mask', util.tensor2im(data['mask_composite_object'][0], normalize=False)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])

    im = visuals['synthesized_image']  # The size is opt.fineSize x opt.fineSize x 3. The range is 0-255.
    mask = visuals['input_mask']  # The size should be opt.fineSize x opt.fineSize x 3, and the range should be 0-255.
    mask = mask[:, :, 0]  # Only use the first chance and reduce the size to opt.fineSize x opt.fineSize x 1.

    im = scipy.misc.imresize(im, [512, 512])
    mask = scipy.misc.imresize(mask, [512, 512])

    im = im.astype(float)
    mask = mask.astype(float)

    im = im[:, :, ::-1]
    im -= np.array((104.00699, 116.66877, 122.67892))
    im = im.transpose((2, 0, 1))

    mask -= 128.0
    mask = mask[np.newaxis, ...]

    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im

    net.blobs['mask'].reshape(1, *mask.shape)
    net.blobs['mask'].data[...] = mask

    # run net for prediction
    net.forward()
    out = net.blobs['output-h'].data[0]
    out = out.transpose((1, 2, 0))
    out += np.array((104.00699, 116.66877, 122.67892))
    out = out[:, :, ::-1]

    neg_idx = out < 0.0
    out[neg_idx] = 0.0
    pos_idx = out > 255.0
    out[pos_idx] = 255.0

    result = out.astype(np.uint8)
    visuals['synthesized_image'] = result
    img_path = data['path']
    print('process image... {}:{}'.format(i, img_path))
    visualizer.save_images(webpage, visuals, img_path)

print('saving')
webpage.save()
