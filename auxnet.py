from __future__ import print_function
import numpy as np
np.random.seed(0)  # for reproducibility
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import merge
from keras.layers import Convolution2D, MaxPooling2D
from config import *
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D

def auxstage(bn,auxres,auxstride,auxlevel,inp,nb_filters,kernel_size,firstcall=1,layerscnt=0):
    # Function auxstage: create an auxstage, use recursion to build the fractals
    # Inputs:
    # bn: Batch normalization , 0: no batch normalization, 1: use batch normalization
    # auxres: #1=Aux with concat #0=Aux with sum #2=resnet style shortcut
    # auxstride: how many stages in deep branch covered by the aux branch
    # auxlevel: how many nested levels
    # nb_filters: List of size auxstride contains filter count for each level
    # inp: the input tesnor
    # Outputs:
    # x: the output tensor
    debug=False
    filtercnt=nb_filters[0]
    subfilters=[filtercnt for _ in range(auxstride-1)]
    if auxres!=2:
        y=Convolution2D(filtercnt, kernel_size[0], kernel_size[1],	border_mode='same')(inp)
        if bn==1:
            y=BatchNormalization(axis=1 if K.image_dim_ordering() == 'th' else -1)(y)
        y =  Activation('relu')(y)
        if firstcall==1:
            for _ in range(auxstride-2):
                y=MaxPooling2D()(y)
    if debug: print('after shallow','firstcall',firstcall,'y.shape',y.get_shape(),'auxlevel',auxlevel,'subfilters',subfilters)
    if auxlevel==1:
        x = Convolution2D(filtercnt*2 if auxres==2 else filtercnt, kernel_size[0], kernel_size[1],	border_mode='same')(inp)
        layerscnt+=1
        if bn==1:
            x=BatchNormalization(axis=1 if K.image_dim_ordering() == 'th' else -1)(x)
        x =  Activation('relu')(x)
    else:
        x=auxstage(bn,auxres,auxstride,auxlevel-1,inp,subfilters,kernel_size,0)
        if firstcall==1:
            x=MaxPooling2D()(x)
            x=Dropout(dropratio)(x)
    for i in range(2,auxstride):
        if auxlevel==1:
            x = Convolution2D(filtercnt*2 if auxres==2 else filtercnt, kernel_size[0], kernel_size[1],	border_mode='same')(x)
            layerscnt+=1
            if bn==1:
                x=BatchNormalization(axis=1 if K.image_dim_ordering() == 'th' else -1)(x)
            x =  Activation('relu')(x)
        else:
            sfilters=[nb_filters[i-1 if firstcall==1 else 0] for _ in range(auxlevel-1)]
            x=auxstage(bn,auxres,auxstride,auxlevel-1,x,sfilters,kernel_size,0)
            if firstcall==1 and i<(auxstride-1):
                x=MaxPooling2D()(x)
                x=Dropout(dropratio)(x)
    if debug: print('before concat','firstcall',firstcall,'y.shape',y.get_shape(),'x.shape',x.get_shape(),'auxlevel',auxlevel,'subfilters',subfilters)
    if auxres==1: #1=Aux #0=Auxsum
        x= merge([x,  y],mode='concat',concat_axis=1 if K.image_dim_ordering() == 'th' else -1)
    elif auxres==0: #Auxsum
        x = merge([x, y], mode='sum')#,axis=1 if K.image_dim_ordering() == 'th' else -1)
    elif auxres==2: #resnet
        x = merge([x, y], mode='sum')#,axis=1 if K.image_dim_ordering() == 'th' else -1)
    #x=BatchNormalization(axis=1 if K.image_dim_ordering() == 'th' else -1)(x)
    print(layerscnt)
    return x
def tail(x,nb_classes):
    if gap:
        x=GlobalAveragePooling2D()(x)
    else:
        x = Convolution2D(nb_filters*2 if auxres==2 else nb_filters[0], kernel_size[0], kernel_size[1])(x)
        if bn==1:
            x=BatchNormalization(axis=1 if K.image_dim_ordering() == 'th' else -1)(x)
        x = Activation('relu')(x)
        #x=MaxPooling2D()(x)
        x=Flatten()(x)
        x=Dense(fc1cnt)(x)
        x=BatchNormalization(axis=1 if K.image_dim_ordering() == 'th' else -1)(x)
        x=Activation('relu')(x)
        x=Dropout(dropratio)(x)
        x=Dense(fc2cnt)(x)
        x=BatchNormalization(axis=1 if K.image_dim_ordering() == 'th' else -1)(x)
    x=Activation('relu')(x)
    x=Dense(nb_classes)(x)
    x=Activation('softmax')(x)
    return x
    
