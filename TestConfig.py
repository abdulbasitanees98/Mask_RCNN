# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:21:13 2020

@author: basit
"""

import config as cfg
from mrcnn2 import model


# define the test configuration
class TestConfig(cfg.Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

# define the model
rcnn = model.MaskRCNN(mode='inference', model_dir='./', config=TestConfig())

# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)