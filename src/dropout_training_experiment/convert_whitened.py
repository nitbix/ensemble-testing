#!/usr/bin/python

from toupee.data import load_data
import cPickle
import numpy as np
import sys

fileName = sys.argv[1]
with open('train.pkl') as f:
    train = cPickle.load(f)
with open('test.pkl') as f:
    test = cPickle.load(f)

train_x = train.X[:40000]
train_y = train.y[:40000].flatten()
valid_x = train.X[40000:]
valid_y = train.y[40000:].flatten()
test_x = test.X
test_y = test.y.flatten()

np.savez_compressed(fileName + 'train',x=train_x,y=train_y)
np.savez_compressed(fileName + 'valid',x=valid_x,y=valid_y)
np.savez_compressed(fileName + 'test',x=test_x,y=test_y)
