#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:30:27 2017

@author: nady
"""

batch_size = 64
nb_epoch = 100
# number of convolutional filters to use in the deep branch
nb_filters = 128
#ctpbase='/work/cascades/nady/data/ctp'
ctpbase='/media/nady/OS/text_detection/coco/COCO-Text-Patch'
# number of convolutional filters to use in the shallow branch
nb_faux = 16
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# Parameters
bn=1;auxres=1;auxname='Aux';auxstride=4;auxlevel=4
#count in FC layers
fc1cnt=512
fc2cnt=512