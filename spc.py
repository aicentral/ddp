#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:50:53 2017

@author: nady
"""

import numpy as np
np.random.seed(0)  # for reproducibility
import os
from helpers import loaddata,createcallbacks,ctpval,ctpgen,prepareimg
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Flatten,Input, merge
from keras.layers import Convolution2D, MaxPooling2D,Dropout
from  skimage.io import imsave,imread
from skimage.transform import resize
from skimage import morphology,measure
from skimage.draw import polygon_perimeter
import matplotlib.pyplot as plt
import pickle
import time
#from keras.utils import np_utils
#from keras.layers.normalization import BatchNormalization
from config import kernel_size,nb_filters
#************************************************************************
#************************************************************************
testonly=True
#************************************************************************
#************************************************************************
expname='spc_linear'
if not os.path.exists(expname):
    os.makedirs(expname)
    os.makedirs('%s/log'%(expname))
if not os.path.exists('./output'):
    os.makedirs('./output')    

K.clear_session()
def mymodel(img_input):
    x=Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same')(img_input)
    x = Activation('relu')(x)
    for _ in range(1):
        x=Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same')(x)
        x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x= Dropout(0.5)(x)
    for i in range(4):
        for _ in range(2):
            x = Convolution2D(nb_filters if i<3 else 1,
                              kernel_size[0], kernel_size[1],border_mode='same')(x)
            x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        if i<3:
            x= Dropout(0.5)(x)
    x = Activation('linear')(x) #try also hard_sigmoid
    x=Flatten()(x)
    return x
def mymodelprint(img_input):
    x=Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same')(img_input)
    for _ in range(1):
        x=Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same')(x)
    x = MaxPooling2D()(x)
    for i in range(4):
        for _ in range(2):
            x = Convolution2D(nb_filters if i<3 else 1,
                              kernel_size[0], kernel_size[1],border_mode='same')(x)
        x = MaxPooling2D()(x)
    return x

if not testonly:
    X_test,Y_test=ctpval(2048)
    g=ctpgen(1024)
    img_input = Input(shape=(32,32,3))#input_shape)
    x=mymodel(img_input)
    model=Model(img_input, x)
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop')
    print(model.summary())  #sample per epoch=15125
    [csv,chkpoint]=createcallbacks(expname)
    model.fit_generator(g, samples_per_epoch=242000, nb_epoch=100,max_q_size=100,
              verbose=1, validation_data=(X_test, Y_test),callbacks=[csv,chkpoint])


###### testing ############
testctp=False
if testctp:
    img_input = Input(shape=(32,32,3))#input_shape)
    g=ctpgen(1024)
    x=mymodel(img_input)
    model=Model(img_input, x)
    print(model.summary())
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop')
    try:
      model.load_weights("./%s/weights.hdf5"%(expname))
      print('****************** pretrained models loaded *************************')
    except:
      print('****************** pretrained ^^^^NOT^^^^ loaded *************************')
      pass
    cnt=0
    correct=0
    for X_test,Y_test in g:
        cnt+=1
        o=model.predict(X_test, batch_size=32, verbose=1)
        for predicted,label in zip(o,Y_test):
            output=0.0 if predicted<0.5 else 1.0
            correct=correct+1 if output==label else correct+0
        if cnt==236:
            break
    print('Accuracy=',correct/2420.0) # ('Accuracy=', 94.699173553719)
############## Test big image sliding
testbigimage=False
if testbigimage:
    (w,h)=(32,32)
    img_input = Input(shape=(w,h,3))#input_shape)
    x=mymodel(img_input)
    model=Model(img_input, x)
    print(model.summary())
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop')
    try:
      model.load_weights("./%s/weights.hdf5"%(expname))
      print('****************** pretrained models loaded *************************')
    except:
      print('****************** pretrained ^^^^NOT^^^^ loaded *************************')
      pass

    #import glob
    #imgs= glob.glob("./testimgs/*.jpg")
    imgs=['a.jpg']#,'stop.jpg']
    start = time.clock()
    for i,fname in zip(range(len(imgs)),imgs):
        x=prepareimg(fname,0,0)
        output=np.zeros((38,38),dtype=int)
        for i in range(0,612-32,16):
            for j in range(0,612-32,16):
                o=model.predict(x[:,i:(i+32),j:(j+32),:], batch_size=1, verbose=0)
                o=0 if o<0.9 else 1
                output[i/16,j/16]=o
        elapsed = (time.clock() - start)
        print ('processing time',elapsed)
        output=morphology.opening(output)
        output=morphology.closing(output)        
        outf32=output.astype(np.float32)
        outf32=resize(outf32,(612,612))
        #l=measure.label(output);    print(l)
        def thresh(x):
            if x>0:
                return 1
            else:
                return 0
        outf32=np.asarray(map(thresh,outf32.reshape(612*612))).reshape((612,612))
        loc=np.where(outf32==1)
        (minr,minc,maxr,maxc)=(np.min(loc[0]),np.min(loc[1]),np.max(loc[0]),np.max(loc[1]))
        rr, cc = polygon_perimeter([minr, minr, maxr, maxr],
                                   [minc, maxc, maxc, minc],
                                   shape=outf32.shape, clip=False)
        out=imread(fname)
        #x[0,rr,cc,:]=-1
        out[rr,cc,:]=255
        plt.imshow(output, cmap='hot', interpolation='nearest')
        plt.show()
        imsave("./output/%s"%fname[-10:],out)

# Test big image conv.
testbigimage=False
if testbigimage:
    (w,h)=(256,256)
    img_input = Input(shape=(w,h,3))#input_shape)
    x=mymodel(img_input)
    model=Model(img_input, x)
    print(model.summary())
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop')
    try:
      model.load_weights("./%s/weights.hdf5"%(expname))
      print('****************** pretrained models loaded *************************')
    except:
      print('****************** pretrained ^^^^NOT^^^^ loaded *************************')
      pass
    #X_test,Y_test=ctpval(1)
    import glob
    imgs= glob.glob("./testimgs/*.jpg")
    imgs=['a.jpg','stop.jpg']
    for i,fname in zip(range(len(imgs)),imgs):
        x=prepareimg(fname,w,h)
        o=model.predict(x, batch_size=1, verbose=1)
        o=np.reshape(o,(w/32,h/32))
        print(o)
        o=o-np.mean(o)
        o=o/np.max(o)
        #o=resize(o, (w,h))
        plt.imshow(o, cmap='hot', interpolation='nearest')
        plt.show()
        imsave("./output/%s"%fname,o)
#    print(o)
    #pickle.dump( o, open( "./output/output_%d.p"%i, "wb" ))
##### save the model graph to png file
printgraph=True
if printgraph:
    from keras.utils.visualize_util import plot
    (w,h)=(32,32)
    img_input = Input(shape=(w,h,3))#input_shape)
    x=mymodelprint(img_input)
    model=Model(img_input, x)
    plot(model, to_file='model.png')