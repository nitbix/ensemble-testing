#!/usr/bin/python
"""
Representation of a Multi-Layer Perceptron

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time
import copy
import numpy
import scipy

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams                                                                                                                    
from toupee.logistic_sgd import LogisticRegression
from toupee import data
from toupee.data import Resampler, Transformer, sharedX
from toupee import update_rules
from toupee import layers
from toupee import config 
from toupee import cost_functions
from toupee.mlp import MLP

def test_mlp(dataset, params, pretraining_set=None, x=None, y=None):
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = (None,None)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / params.batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / params.batch_size

    print '... building the model'

    index = T.lscalar()
    if x is None:
        x = T.matrix('x')
    if y is None:
        y = T.ivector('y')

    rng = numpy.random.RandomState(params.random_seed)

    classifier = MLP(params=params, rng=rng, input=x, index=index, x=x, y=y,
            pretraining_set=pretraining_set)

    if len(dataset) > 2:
        test_set_x, test_set_y = dataset[2]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / params.batch_size

    def make_models(classifier):
        validate_model = classifier.eval_function(index,valid_set_x,valid_set_y,x,y)
        train_model = classifier.train_function(index,train_set_x,train_set_y,x,y)
        if len(dataset) > 2:
            test_model = classifier.eval_function(index,test_set_x,test_set_y,x,y)
        else:
            test_model = None
        return (train_model, validate_model, test_model)

    print '... {0} training'.format(params.training_method)

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 20  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.99999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    start_time = time.clock()
    best_classifier = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    epoch = 0
    done_looping = False
    previous_minibatch_avg_cost = 1
    if params.training_method == 'normal':
        train_model, validate_model, test_model = make_models(classifier)
        while (epoch < params.n_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index,previous_minibatch_avg_cost)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                         (epoch, minibatch_index + 1, n_train_batches,
                          this_validation_loss * 100.))

                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        best_classifier = copy.copy(classifier)

                        # test it on the test set
                        if test_model is not None:
                            test_losses = [test_model(i) for i
                                           in xrange(n_test_batches)]
                            test_score = numpy.mean(test_losses)

                            print(('     epoch %i, minibatch %i/%i, test error of '
                                   'best model %f %%') %
                                  (epoch, minibatch_index + 1, n_train_batches,
                                   test_score * 100.))

                if patience <= iter:
                        print('finished patience')
                        done_looping = True
                        break
    elif params.training_method == 'greedy':
        all_layers = classifier.hiddenLayers
        for l in xrange(len(all_layers)):
            best_classifier = None
            best_validation_loss = numpy.inf
            best_iter = 0
            test_score = 0.
            epoch = 0
            done_looping = False
            print "training {0} layers\n".format(l + 1)
            classifier.hiddenLayers = all_layers[:l+1]
            classifier.make_top_layer(params.n_out,classifier.hiddenLayers[l].output,
                    classifier.hiddenLayers[l].output_shape,rng)
            train_model, validate_model, test_model = make_models(classifier)
            while (epoch < params.n_epochs) and (not done_looping):
                epoch = epoch + 1
                for minibatch_index in xrange(n_train_batches):

                    minibatch_avg_cost = train_model(minibatch_index,previous_minibatch_avg_cost)
                    iter = (epoch - 1) * n_train_batches + minibatch_index

                    if (iter + 1) % validation_frequency == 0:
                        validation_losses = [validate_model(i) for i
                                             in xrange(n_valid_batches)]
                        this_validation_loss = numpy.mean(validation_losses)

                        print('epoch %i, minibatch %i/%i, validation error %f %%' %
                             (epoch, minibatch_index + 1, n_train_batches,
                              this_validation_loss * 100.))

                        if this_validation_loss < best_validation_loss:
                            if this_validation_loss < best_validation_loss *  \
                                   improvement_threshold:
                                patience = max(patience, iter * patience_increase)

                            best_validation_loss = this_validation_loss
                            best_iter = iter
                            best_classifier = classifier

                            # test it on the test set
                            if test_model is not None:
                                test_losses = [test_model(i) for i
                                               in xrange(n_test_batches)]
                                test_score = numpy.mean(test_losses)

                                print(('     epoch %i, minibatch %i/%i, test error of '
                                       'best model %f %%') %
                                      (epoch, minibatch_index + 1, n_train_batches,
                                       test_score * 100.))

                    if patience <= iter:
                            print('finished patience')
                            done_looping = True
                            break
            classifier = best_classifier
    end_time = time.clock()
    if test_set_x is not None:
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        return classifier
    else:
        print('Selection : Best validation score of {0} %'.format(
              best_validation_loss * 100.))
        return best_classifier

if __name__ == '__main__':
    #turn this on only if you want to do parameter search
    search_epochs = 40
    search = False

    params = config.load_parameters(sys.argv[1])
    dataset = data.load_data(params.dataset,
                              shared = True,
                              pickled = params.pickled)
    pretraining_set = data.make_pretraining_set(dataset,params.pretraining)
    if not search:
        mlp=test_mlp(dataset, params, pretraining_set = pretraining_set)
    else:
        params.n_epochs = search_epochs
        for eta_minus in [0.01,0.1,0.5,0.75,0.9]:
            params.update_rule.eta_minus = eta_minus
            for eta_plus in [1.001,1.01,1.1,1.2,1.5]:
                params.update_rule.eta_plus = eta_plus
                for min_delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
                    params.update_rule.min_delta = min_delta
                    for max_delta in [50]:
                        print "PARAMS:"
                        print "ETA-: {0}".format(eta_minus)
                        print "ETA+: {0}".format(eta_plus)
                        print "MIN_DELTA: {0}".format(min_delta)
                        print "MAX_DELTA: {0}".format(max_delta)
                        params.update_rule.max_delta = max_delta
                        try:
                            mlp=test_mlp(dataset, params, pretraining_set = pretraining_set)
                        except KeyboardInterrupt:
                            print "skipping manually to next"
