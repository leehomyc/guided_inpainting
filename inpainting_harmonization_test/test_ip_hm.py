"""This is the file to test inpainting and harmonization jointly. The input
is an image pair, where one image is the foreground object, which we have to
specify the annotation ID of the object. The second image is the background,
which we have to specify the new location of the upper left corner and the
height and width to inpaint the object. Please refer to
inpainting_dataset_test.py for reference. """

from collections import OrderedDict
import os

# Files related to segmentation
import third_party.Mask_RCNN.model as modellib
import third_party.Mask_RCNN.coco as coco

from inpainting_harmonization_test.data.inpainting_dataset_test import \
    InpaintingDatasetTest
from inpainting.models.models import create_model as create_inpainting_model
from harmonization.models.models import \
    create_model as create_harmonization_model
from inpainting_harmonization_test.options.test_options import TestOptions
from inpainting_harmonization_test.util import html
import inpainting_harmonization_test.util.util as util
from inpainting_harmonization_test.util.visualizer import Visualizer


def parse_options():
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    return opt


def build_inpainting_model(opt):
    opt.model = 'inpainting_guided'
    opt.checkpoints_dir = opt.ip_checkpoints_dir
    model_inpainting = create_inpainting_model(opt)
    return model_inpainting


def build_harmonization_model(opt):
    opt.model = 'inpainting_harm'
    opt.checkpoints_dir = opt.harm_checkpoints_dir
    opt.interpolated_conv = False
    opt.dilation = 1
    model_harmonization = create_harmonization_model(opt)
    return model_harmonization


def create_webpage_visualizer(opt):
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))
    return visualizer, webpage


def create_segmentation_model(opt):
    model = None

    if opt.use_segmentation is True:
        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        model = modellib.MaskRCNN(mode="inference",
                                  model_dir='{}/log'.format(opt.seg_model_path),
                                  config=InferenceConfig())
        model.load_weights('{}/mask_rcnn_coco.h5'.format(opt.seg_model_path),
                           by_name=True)
    return model


if __name__ == '__main__':
    opt = parse_options()
    model_inpainting = build_inpainting_model(opt)
    model_harmonization = build_harmonization_model(opt)
    visualizer, webpage = create_webpage_visualizer(opt)
    seg_model = create_segmentation_model(opt)

    InpaintingDatasetTest = InpaintingDatasetTest()
    InpaintingDatasetTest.initialize(opt, seg_model)

    for i in range(opt.how_many):
        if i >= opt.how_many:
            break
        print('processing image {}'.format(i))
        data = InpaintingDatasetTest.get_item(i)
        generated = model_inpainting.inference(data['input'], data['mask'])
        visuals = OrderedDict(
            [('input',
              util.tensor2label(data['image_composite_with_bg'][0],
                                opt.label_nc)),
             # noqa 501
             ('mask',
              util.tensor2im(data['mask_composite_object'][0],
                             normalize=False)),
             # noqa 501
             ('inpainting', util.tensor2im(generated.data[0]))])  # noqa 501
        img_path = data['path']
        inpainted_image = generated.data[0]
        inpainted_image = inpainted_image[None, :, :, :]

        generated = model_harmonization.inference(inpainted_image, data[
            'mask_composite_object'])  # noqa 501

        visuals['inpainting_harmonization'] = util.tensor2im(generated.data[0])
        visualizer.save_images(webpage, visuals, img_path)

    print('saving')
    webpage.save()
