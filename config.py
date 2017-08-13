#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:30:27 2017

@author: nady
"""

batch_size = 16
nb_epoch = 10
# number of convolutional filters to use in the deep branch
#nb_filters = [64,128,256,512]
nb_filters = [4,8,16,32,64,128]
#nb_filters = [16,32,64,256,256]
#nb_filters = [32,64,128,256,512]
# number of convolutional filters to use in the shallow branch
nb_faux = nb_filters[0]
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# Parameters
bn=0;
auxres=1;#1=Aux #0=Auxsum
auxname='Aux';auxstride=3;auxlevel=3
#count in FC layers
gap=True
fc1cnt=4096
fc2cnt=4096
dropratio=0.5
