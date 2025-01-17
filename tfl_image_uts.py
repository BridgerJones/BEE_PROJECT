########################################################
# module: tfl_image_anns.py
# authors: vladimir kulyukin
# descrption: Unit tests for Project 1 image ANN and ConvNet
# trained with TFLearn.
# bugs to vladimir kulyukin in canvas
# to install tflearn to go http://tflearn.org/installation/
########################################################

from tfl_image_anns import *
from tfl_image_convnets import *
import tensorflow as tf
import unittest

class tfl_image_uts(unittest.TestCase):

    def test_tfl_image_ann_bee1_gray(self):
        tf.compat.v1.reset_default_graph()
        ian = load_image_ann_model('models/img_ann.tfl')
        vacc = validate_tfl_image_ann_model(ian, BEE1_gray_valid_X, BEE1_gray_valid_Y)
        print('**** Ann valid. acc on BEE1_gray = {}'.format(vacc))

    def test_tfl_image_ann_bee2_2s_gray(self):
        tf.compat.v1.reset_default_graph()
        ian = load_image_ann_model('models/img_ann.tfl')
        vacc = validate_tfl_image_ann_model(ian, BEE2_1S_gray_valid_X, BEE2_1S_gray_valid_Y)
        print('**** Ann valid. acc on BEE2_1S_gray = {}'.format(vacc))

    def test_tfl_image_ann_bee4_gray(self):
        tf.compat.v1.reset_default_graph()
        ian = load_image_ann_model('models/img_ann.tfl')
        vacc = validate_tfl_image_ann_model(ian, BEE4_gray_valid_X, BEE4_gray_valid_Y)
        print('**** Ann valid. acc on BEE4_gray = {}'.format(vacc))

    def test_tfl_image_convnet_bee1(self):
        tf.compat.v1.reset_default_graph()
        img_cn = load_image_convnet_model('models/img_cn.tfl')
        vacc = validate_tfl_image_convnet_model(img_cn, BEE1_valid_X, BEE1_valid_Y)
        print('**** ConvNet valid. acc on BEE1 = {}'.format(vacc))

    def test_tfl_image_convnet_bee2_1s(self):
        tf.compat.v1.reset_default_graph()
        img_cn = load_image_convnet_model('models/img_cn.tfl')
        vacc = validate_tfl_image_convnet_model(img_cn, BEE2_valid_X, BEE2_valid_Y)
        print('**** ConvNet valid. acc on BEE2_1S = {}'.format(vacc))

    def test_tfl_image_convnet_bee4(self):
        tf.compat.v1.reset_default_graph()
        img_cn = load_image_convnet_model('models/img_cn.tfl')
        vacc = validate_tfl_image_convnet_model(img_cn, BEE4_valid_X, BEE4_valid_Y)
        print('**** ConvNet valid. acc on BEE4 = {}'.format(vacc))

### ================ Unit Tests ====================

if __name__ == '__main__':
    unittest.main()
