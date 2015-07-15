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

import numpy

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams                                                                                                                    
from logistic_sgd import LogisticRegression
import data
from data import Resampler, Transformer, sharedX
import update_rules
import layers

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: (int,int) array
        :param n_hidden: number of hidden units and dropout rate for each hidden
        layer

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        #TODO:
        # - rename *n_in to *input_shape
        # - flatten shapes for flat layer

        self.hiddenLayers = []
        chain_n_in = n_in
        chain_in = input
        first_layer = True
        prev_dim = None
        for layer_type,desc in n_hidden:
            if(layer_type == 'flat'):
                n_this,drop_this,name_this,activation_this = desc
                l = layers.FlatLayer(rng=rng, inputs=chain_in.flatten(ndim=2),
                                n_in=numpy.prod(chain_n_in), n_out=numpy.prod(n_this),
                                activation=activation_this,dropout_rate=drop_this,
                                layer_name=name_this)
                chain_n_in = n_this
                chain_in=l.output
                self.hiddenLayers.append(l)
                first_layer = False
            if(layer_type == 'conv'):
                input_shape,filter_shape,pool_size,drop_this,name_this,activation_this = desc
                if prev_dim is None:
                    prev_dim = (input_shape[0],input_shape[1],input_shape[2])
                l = layers.ConvolutionalLayer(rng=rng,
                                       inputs=chain_in, 
                                       input_shape=input_shape, 
                                       filter_shape=filter_shape,
                                       pool_size=pool_size,
                                       activation=activation_this,
                                       dropout_rate=drop_this,
                                       layer_name = name_this)
                prev_map_number,dim_x,dim_y = prev_dim
                curr_map_number = filter_shape[0]
                output_dim_x = (dim_x - filter_shape[2] + 1) / pool_size[0]
                output_dim_y = (dim_y - filter_shape[3] + 1) / pool_size[1]
                chain_n_in = (curr_map_number,dim_x,dim_y)
                prev_dim = (curr_map_number,dim_x,dim_y)
                chain_in = l.output
                self.hiddenLayers.append(l)
                first_layer = False

        # The logistic regression layer gets as input the hidden units
        # of the last hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=chain_in,
            n_in=chain_n_in,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = sum([abs(hiddenLayer.W).sum()
                    for hiddenLayer in self.hiddenLayers]) \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = sum([(hiddenLayer.W ** 2).sum() for hiddenLayer in
                        self.hiddenLayers]) \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        self.errors = self.logRegressionLayer.errors
        self.y_pred = self.logRegressionLayer.y_pred

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        p = self.logRegressionLayer.params
        for hiddenLayer in self.hiddenLayers:
            p += hiddenLayer.params
        self.params = p


def test_mlp(datasets,learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             batch_size=20, n_hidden=(500,0), update_rule=update_rules.sgd, n_in=28*28):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=n_in,
                     n_hidden=n_hidden, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    previous_cost = T.lscalar()
    updates = []
    theano_rng = MRG_RandomStreams(max(rng.randint(2 ** 15), 1))

    dropout_rates = {}
    for layer in classifier.hiddenLayers:
        dropout_rates[layer.layer_name + '_W'] = layer.dropout_rate
    for param, gparam in zip(classifier.params, gparams):
        if param in dropout_rates:
            include_prob = 1 - dropout_rates[param]
        else:
            include_prob = 1
        mask = theano_rng.binomial(p=include_prob,
                                   size=param.shape,dtype=param.dtype)    
        new_update = update_rule(param, learning_rate, gparam, mask, updates,
                cost,previous_cost)
        updates.append((param, new_update))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index,previous_cost],
            outputs=cost,
            on_unused_input='warn',
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

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

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    previous_minibatch_avg_cost = 1
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index,previous_minibatch_avg_cost)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
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

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return classifier

def train_and_select(x,y,training_set, validation_set, learning_rate=0.01,
                     L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
                     batch_size=20, n_hidden=(500,0),
                     update_rule=update_rules.sgd,n_in=28*28):
    """
    Train a classifier and select the version with the best validation
    error
    """
    train_set_x, train_set_y = training_set
    valid_set_x, valid_set_y = validation_set

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=n_in,
                     n_hidden=n_hidden, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    previous_cost = T.lscalar()
    updates = []
    theano_rng = MRG_RandomStreams(max(rng.randint(2 ** 15), 1))

    dropout_rates = {}
    for layer in classifier.hiddenLayers:
        dropout_rates[layer.layer_name + '_W'] = layer.dropout_rate
    for param, gparam in zip(classifier.params, gparams):
        if param in dropout_rates:
            include_prob = 1 - dropout_rates[param]
        else:
            include_prob = 1
        mask = theano_rng.binomial(p=include_prob,
                                   size=param.shape,dtype=param.dtype)    
        new_update = update_rule(param, learning_rate, gparam, mask, updates,
                cost,previous_cost)
        updates.append((param, new_update))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index,previous_cost],
            outputs=cost,
            on_unused_input='ignore',
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]
            })
    ###############
    # TRAIN MODEL #
    ###############

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.99  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_classifier = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    previous_minibatch_avg_cost = 1
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index,previous_minibatch_avg_cost)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    best_classifier = classifier

                    print("\repoch %i, minibatch %i/%i, validation error %f %%" %
                         (epoch, minibatch_index + 1, n_train_batches,
                          this_validation_loss * 100.))

#                    print(('     epoch %i, minibatch %i/%i, test error of '
#                           'best model %f %%') %
#                          (epoch, minibatch_index + 1, n_train_batches,
#                           test_score * 100.))

            if patience <= iter:
                    print('finished patience')
                    done_looping = True
                    break

    end_time = time.clock()
    print('Selection : Best validation score of {0} %'.format(
          best_validation_loss * 100.))
    return best_classifier


if __name__ == '__main__':
    ###parameters###
    dataset_name = 'cifar10'
    L1_reg=0.00
    L2_reg=0.00
    n_epochs=500
    search_epochs = 40
    transform = False
    batch_size=300
    update_rule=update_rules.rprop
    search = False

    if dataset_name == 'mnist':
        learning_rate=0.01
        eta_plus = 1.01
        eta_minus = 0.1
        max_delta = 5
        min_delta = 1e-3
        dataset='/local/mnist.pkl.gz'
        pickled=True
        n_hidden=[
                  ('flat',(2500,0.5,'h0',T.tanh)),
                  ('flat',(2000,0.5,'h1',T.tanh)),
                  ('flat',(1500,0.5,'h2',T.tanh)),
                  ('flat',(1000,0.5,'h2',T.tanh)),
                  ('flat',(500,0.5,'h3',T.tanh))
                 ]
        n_in = 784
    elif dataset_name == 'mnist-transformed':
        learning_rate=0.01
        eta_plus = 1.1
        eta_minus = 0.01
        max_delta = 5
        min_delta = 1e-3
        dataset='/local/mnit-transformed/'
        pickled=False
        n_hidden=[
                  ('flat',(2500,0.5,'h0',T.tanh)),
                  ('flat',(2000,0.5,'h1',T.tanh)),
                  ('flat',(1500,0.5,'h2',T.tanh)),
                  ('flat',(1000,0.5,'h2',T.tanh)),
                  ('flat',(500,0.5,'h3',T.tanh))
                 ]
        n_in = 784
    elif dataset_name == 'cifar10':
        learning_rate=0.001
        dataset='/local/cifar10/'
        pickled=False
        n_hidden=[
                  #input_shape,filter_shape,pool_size,drop_this,name_this,activation_this
                  ('conv',([batch_size,3,32,32],[5,3,10,10],[4,4],0.5,'c1',T.tanh)),
                  ('flat',(3000,0.5,'f0',T.tanh))
                 ]
        n_in = 3072
    else:
        print "unknown dataset_name " + dataset_name

    ###parameters end###

    datasets = data.load_data(dataset, shared = not transform, pickled = pickled)

    for arg in sys.argv[1:]:
        if arg[0]=='-':
            exec(arg[1:])
    if not search:
        def update_rule(param,learning_rate,gparam,mask,updates,
                        current_cost,previous_cost):
            return update_rules.rprop(param,learning_rate,gparam,mask,updates,
                                      current_cost,previous_cost,
                                      eta_plus=eta_plus,eta_minus=eta_minus,
                                      max_delta=max_delta,min_delta=min_delta)
        mlp=test_mlp(datasets,learning_rate, L1_reg, L2_reg, n_epochs,
            batch_size, n_hidden, update_rule = update_rules.old_rprop, n_in = n_in)
    else:
        for eta_minus in [0.01,0.1,0.5,0.75,0.9]:
            for eta_plus in [1.001,1.01,1.1,1.2,1.5]:
                for min_delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
                    for max_delta in [50]:
                        print "PARAMS:"
                        print "ETA-: {0}".format(eta_minus)
                        print "ETA+: {0}".format(eta_plus)
                        print "MIN_DELTA: {0}".format(min_delta)
                        print "MAX_DELTA: {0}".format(max_delta)
                        def update_rule(param,learning_rate,gparam,mask,updates,current_cost,previous_cost):
                            return rprop(param,learning_rate,gparam,mask,updates,current_cost,previous_cost,
                                         eta_plus=eta_plus,eta_minus=eta_minus,max_delta=max_delta,min_delta=min_delta)
                        try:
                            n_epochs = search_epochs
                            mlp=test_mlp(datasets,learning_rate, L1_reg, L2_reg, n_epochs, batch_size,
                                         n_hidden, update_rule = update_rule, n_in = n_in)
                        except KeyboardInterrupt:
                            print "skipping manually to next"
