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
    members = []
    for i in range(0,params.ensemble_size):
        print 'training member {0}'.format(i)
        new_member = method.create_member(x,y)
        members.append(new_member)
        gc.collect()
    ensemble = params.method.create_aggregator(params,members,x,y,train_set,valid_set)
    if len(sys.argv) > 2:
        dill.dump(method.members,open(sys.argv[2],"wb"))
    test_set_x, test_set_y = method.resampler.get_test()
    test_model = theano.function(inputs=[index],
            outputs=ensemble.errors,
            givens={
                x: test_set_x[index * params.batch_size:(index + 1) *
                    params.batch_size],
                y: test_set_y[index * params.batch_size:(index + 1) *
                    params.batch_size]})
    n_test_batches = test_set_x.shape[0].eval() / params.batch_size
    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_score = numpy.mean(test_losses)
    print 'Final error: {0} %'.format(test_score * 100.)
