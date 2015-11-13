#!/usr/bin/python

import data
import sys
import cPickle
import gzip
import os
import copy
import math
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) > 2:
        dataset = sys.argv[1]
        fileName = sys.argv[2]
    else:
        raise Exception("need source and destination")
    dataset = data.load_data(dataset,pickled=False,shared=False)
    train,valid,test = dataset
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test
#    n_in = test_x.shape[1]
#    params = { 'alpha': 0.0,
#               'beta': 30.0,
#               'gamma': 20.0,
#               'sigma': 1,
#               'pflip': 0.0,
#               'translation': 3.0,
#               'bilinear': True 
#             }
#    t = data.GPUTransformer(sharedX(train_x),
#                    x=int(math.sqrt(n_in)),
#                    y=int(math.sqrt(n_in)),
#                    progress=False,
#                    save=True,
#                    opts=params)
#    transformed_train_x = t.get_data()

    instances = train_x.shape[0]
    print train_x.shape
    width = height = int(math.sqrt(train_x.shape[1]/3))
    print width, height
    def flip_x(x):
        c = copy.copy(x).reshape([instances,3,width,height])
        for i in range(0,width/2):
            for j in range(0,height):
                for k in range(0,instances):
                    for l in range(0,3):
                        c[k,l,j,i] = c[k,l,j,width - i - 1]
        return c.reshape([instances,3 * height * width])
    def flip_x(x):
        c = copy.copy(x).reshape([instances,3,width,height])
        for i in range(0,width):
            for j in range(0,height/2):
                for k in range(0,instances):
                    for l in range(0,3):
                        c[k,l,j,i] = c[k,l,height-j - 1,i]
        return c.reshape([instances,3 * height * width])
    flipped_train_x_x = flip_x(train_x)
    flipped_train_x_y = flip_x(train_x)
    flipped_train_x_xy = flip_x(flipped_train_x_x)
    atx = np.concatenate([train_x,flipped_train_x_x,flipped_train_x_y,flipped_train_x_xy])
    aty = np.concatenate([train_y,train_y,train_y,train_y])
    print "... saving"
    np.savez_compressed(fileName + 'train',x=atx,y=aty)
    np.savez_compressed(fileName + 'valid',x=valid_x,y=valid_y)
    np.savez_compressed(fileName + 'test',x=test_x,y=test_y)
