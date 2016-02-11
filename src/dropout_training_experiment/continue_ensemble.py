#!/usr/bin/python

import gc
import sys
import numpy as np
import numpy.random
import theano
import theano.tensor as T
import dill

from toupee import config
from toupee.data import *
from toupee.mlp import test_mlp

if __name__ == '__main__':
    params = config.load_parameters(sys.argv[1])
    dataset = load_data(params.dataset,
                              resize_to = params.resize_data_to,
                              shared = False,
                              pickled = params.pickled)
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar('index')
    method = params.method
    method.prepare(params,dataset)
    train_set = method.resampler.get_train()
    valid_set = method.resampler.get_valid()
    test_set = method.resampler.get_test()
    test_set_x, test_set_y = test_set
    shared_dataset = [train_set,valid_set,test_set]
    continuations = dill.load(open(sys.argv[2]))
    members = []
    i = 0
    for c in continuations:
        print "training member {0}".format(i)
        m = test_mlp(shared_dataset, params, continuation = c, x=x, y=y,
                     index=index)
        members.append(m.get_weights())
        m.clear()
        del m
        gc.collect()
        i += 1
    dill.dump(members,open(sys.argv[3],"wb"))
