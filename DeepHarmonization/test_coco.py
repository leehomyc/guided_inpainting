"""This is to test the COCO files using deep harmonization and compare with our model."""

import caffe
from collections import OrderedDict
import numpy as np
import os
import scipy.misc

from harmonization.data.data_loader import CreateDataLoader
from harmonization.options.test_options import TestOptions
from harmonization.util import util, html
from harmonization.util.visualizer import Visualizer

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))  # noqa 501
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
    opt.name, opt.phase, opt.which_epoch))  # noqa 501

net = caffe.Net('DeepHarmonization/model/deploy_512.prototxt',
                'DeepHarmonization/model/harmonize_iter_200000.caffemodel',
                caffe.TEST)

size = np.array([512, 512])
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    img_path = data['path']
    print('process image... {}:{}'.format(i, img_path))

    im_ori = data['input'][0].numpy()  # change tensor to numpy
    mask = data['mask'][0].numpy()  # change tensor to numpy

    im_ori = np.rollaxis(im_ori, 0, 3)  # change cxhxw to hxwxc
    im = (im_ori + 1) * 122.5  # change to 0-255
    im = scipy.misc.imresize(im, size)
    mask = np.rollaxis(mask, 0, 3)  # change cxhxw to hxwxc
    mask = mask * 255
    mask = scipy.misc.imresize(mask, size)

    im = im.astype('float32')
    mask = mask.astype('float32')

    im = im[:, :, ::-1]  # RGB to BGR
    im -= np.array((104.00699, 116.66877, 122.67892))
    im = im.transpose((2, 0, 1))  # change back to cxhxw

    mask = mask[:, :, 0]
    mask -= 128.0
    mask = mask[np.newaxis, ...]

    # shape for input (data blob is N x C x H x W), set data
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

    # out = scipy.misc.imresize(out, [256, 256])
    visuals = OrderedDict([('input_image', util.tensor2label(data['input'][0], opt.label_nc)),  # noqa 501
                           ('synthesized_image', out.astype(np.uint8))])  # noqa 501

    visualizer.save_images(webpage, visuals, img_path)

print('saving')
webpage.save()
