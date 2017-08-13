import caffe
import lmdb
from StringIO import StringIO
from keras.utils import np_utils
import numpy as np
#batchsize=100
img_rows, img_cols = 256,256
nb_classes = 1000
nb_channels = 3
def read_lmdb(lmdb_file,batchsize):
    timg=np.zeros((batchsize,img_rows,img_cols,nb_channels))
    tlbl=np.zeros((batchsize))
    while True:
        cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
        datum = caffe.proto.caffe_pb2.Datum()
        idx=0
        for _, value in cursor:
            datum.ParseFromString(value)
            s = StringIO()
            s.write(datum.data)
            s.seek(0)
            timg[idx] = caffe.io.datum_to_array(datum).reshape(img_rows, img_cols, nb_channels)
            #print datum.label
            tlbl[idx]=int(datum.label)
            idx+=1
            if idx==batchsize:
                idx=0
                timg -= 120
                timg /= 128.
                #timg = timg
                lbl=np_utils.to_categorical(tlbl, nb_classes)
                #print 'batch out *******************************************'
                yield timg,lbl
        
        cursor.close()
#lmdb_dir = '/home/nady/models/hu/ilsvrc12_val_lmdb/'
#lmdb_dir = '/work/newriver/nady/data/imagenet/ILSVRC2015/ilsvrc12_train_lmdb/'
#for im, label in read_lmdb(lmdb_dir):
#    print label, im.shape
#    if label[0]==500:
#       break
