'''Trains a simple convnet on the CIFAR10 dataset.

'''

from __future__ import print_function
import numpy as np
np.random.seed(0)  # for reproducibility
import os
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard,CSVLogger,ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from auxnet import auxstage,tail
from config import *
import sys
#sys.setrecursionlimit(7000)

########################### TEST ONLY FLAG #############################
testonly=False
########################################################################
#expname='mnist150'  # mnist150-> iterate from 1 to 50 internal layer
#expname='cifar%d%d_%d_%s_sgd'%(auxstride,auxlevel,nb_filters[0],'FC' if gap==False else 'GAP')
expname='c10'
nb_classes = 10

#make sure experiment directory is ready
if not os.path.exists(expname):
    os.makedirs(expname)
    os.makedirs('%s/log'%(expname))
    
# input image dimensions
img_rows, img_cols = 32, 32
nb_channels = 3
#############################################batch size manual change #######################
#batch_size = 16
#############################################batch size manual change #######################
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_test=y_test[:64]
X_test=X_test[:64]
trainingmean=np.mean(X_train,axis=0)
trainingstd=np.std(X_train,axis=0)
X_train=(X_train-trainingmean)/trainingstd
X_test=(X_test-trainingmean)/trainingstd
X_train = X_train.astype('float16')
X_test = X_test.astype('float16')

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], nb_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], nb_channels, img_rows, img_cols)
    input_shape = (nb_channels, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, nb_channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, nb_channels)
    input_shape = (img_rows, img_cols, nb_channels)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-7)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
resultsfile=open('./%s/results_%s.csv'%(expname,auxname),'a',0)
K.clear_session()
img_input = Input(shape=input_shape,name='%s%s%s'%(expname,'BN' if bn==1 else 'NBN',auxname)) # this will help finding the experiment in tensorboard
tb=TensorBoard(log_dir='./%s/log'%(expname), histogram_freq=1, write_graph=True, write_images=False)
csv = CSVLogger('./%s/training_%s_%s.csv'%(expname,'BN' if bn==1 else 'NBN',auxname),append=True)
chkpoint = ModelCheckpoint(filepath="./%s/weights_%d%s_%s.hdf5"%(expname,auxlevel,'BN' if bn==1 else 'NBN',auxname), verbose=1, save_best_only=True)

x=Convolution2D(nb_faux, kernel_size[0], kernel_size[1],	border_mode='same')(img_input)
if bn==1:
    x=BatchNormalization(axis=1 if K.image_dim_ordering() == 'th' else -1)(x)
x =  Activation('relu')(x)

x=auxstage(bn,auxres,auxstride,auxlevel,x,nb_filters,kernel_size)
x=tail(x,nb_classes)
model=Model(img_input, x)

try:
  model.load_weights("./%s/weights_%d%s_%s.hdf5"%(expname,auxlevel,'BN' if bn==1 else 'NBN',auxname))
  print('****************** pretrained models loaded *************************')
except:
  print('****************** pretrained ^^^^NOT^^^^ loaded *************************')
  pass

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
#print('%s_%d'%(expname,layerscnt))
#print('****** Inner Conv Layers count = ',layerscnt, auxname,' **** Aux Stride=',auxstride,' BN' if bn==1 else 'NBN')
paramcnt=model.count_params()

if not testonly:
    resultsfile.write('lvl,stride,filters in deep branch,param cnt,BN,auxname\n')
    resultsfile.write('%d,%d,%d,%d,%d,%s\n'%(auxlevel,auxstride,nb_filters[0],paramcnt,bn,auxname))
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.10,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.10,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    
    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)
    print(model.summary())
    spe=X_train.shape[0] 
    print('using sgd')                
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), samples_per_epoch=spe, nb_epoch=nb_epoch,max_q_size=100,
              verbose=1, validation_data=(X_test, Y_test),callbacks=[csv,chkpoint,lr_reducer])

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
if K.image_dim_ordering() == 'th':
    X_test = X_test.reshape(X_test.shape[0], nb_channels, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, nb_channels)
X_test=(X_test-trainingmean)/trainingstd
X_test = X_test.astype('float16')
Y_test = np_utils.to_categorical(y_test, nb_classes)
score = model.evaluate(X_test, Y_test, verbose=1)
#paramcnt=model.count_params()
#del x, model,score,chkpoint # img_input,,tb,csv,chkpoint
print(expname)
print('done',score)
resultsfile.write('val acc=')
resultsfile.write(','.join(str(e) for e in score))
resultsfile.write('\ntraining ended with early stopper or end of epochs')
print('done',score)
resultsfile.close()
#score = model.evaluate(X_test, Y_test, verbose=1)
#paramcnt=model.count_params()
