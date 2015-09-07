#!/usr/bin/python

import gc
import sys
import numpy as np
import numpy.random
import theano
import theano.tensor as T

import mlp
import config
from logistic_sgd import LogisticRegression
from data import *

if __name__ == '__main__':
    params = config.load_parameters(sys.argv[1])
    dataset = load_data(params.dataset,
                              shared = False,
                              pickled = params.pickled)
    resampler = Resampler(dataset)
    train_set = resampler.get_train()
    valid_set = resampler.get_valid()
    x = T.matrix('x')
    y = T.ivector('y')
    members = []
    for i in range(0,params.ensemble_size):
        print 'training member {0}'.format(i)
        mlp_training_dataset = (resampler.make_new_train(params.resample_size),valid_set)
        pretraining_set = make_pretraining_set(mlp_training_dataset,params.pretraining)
        m = mlp.test_mlp(mlp_training_dataset, params,
                pretraining_set = pretraining_set, x=x, y=y)
        members.append(m)
        gc.collect()
    ensemble = params.method.create(params,members,x,y,train_set,valid_set)
    test_set_x, test_set_y = resampler.get_test()
    test_model = theano.function(inputs=[],
        on_unused_input='warn',
        outputs=ensemble.errors,
        givens={x:test_set_x, y:test_set_y})
    test_score = test_model()
    print 'Final error: {0} %'.format(test_score * 100.)
