import scipy.io as sio
import os

a=sio.loadmat('../../../guided_inpainting_output/index_ade20k.mat')

img_len = len(a['index']['filename'][0][0][0])

filename = os.path.join(a['index']['folder'][0][0][0][0], a['index']['filename'][0][0][0][0])