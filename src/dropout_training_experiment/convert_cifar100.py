#!/usr/bin/python

import os
import sys
import cPickle
import numpy as np

fileName='/local/cifar100/'

def unpickle(file):
    fo = open(file,'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        os.chdir(sys.argv[1])
    data = unpickle('cifar-100-python/train')
    test = unpickle('cifar-100-python/test')

    train_x = data['data'][:40000]
    train_y = data['fine_labels'][:40000]
    valid_x = data['data'][40000:]
    valid_y = data['fine_labels'][40000:]
    test_x = test['data']
    test_y = test['fine_labels']
    print "... saving"
    np.savez_compressed(fileName + 'train',x=train_x,y=train_y)
    np.savez_compressed(fileName + 'valid',x=valid_x,y=valid_y)
    np.savez_compressed(fileName + 'test',x=test_x,y=test_y)
