#!/usr/bin/python
"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

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

rectifier = lambda x: T.maximum(0, x)
softsign = lambda x: x / (1 + abs(x))

def sgd(param,learning_rate,gparam,mask,updates,current_cost,previous_cost):
    return param - learning_rate * gparam * mask

def old_rprop(param,learning_rate,gparam,mask,updates,current_cost,previous_cost,
          eta_plus=1.2,eta_minus=0.5,max_delta=50, min_delta=1e-6):
    previous_grad = sharedX(numpy.ones(param.shape.eval()),borrow=True)
    delta = sharedX(learning_rate * numpy.ones(param.shape.eval()),borrow=True)
    previous_inc = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
    zero = T.zeros_like(param)
    one = T.ones_like(param)
    change = previous_grad * gparam

    new_delta = T.clip(
            T.switch(
                T.gt(change,0.),
                delta*eta_plus,
                T.switch(
                    T.lt(change,0.),
                    delta*eta_minus,
                    delta
                )
            ),
            min_delta,
            max_delta
    )
    new_previous_grad = T.switch(
            T.gt(change,0.),
            gparam,
            T.switch(
                T.lt(change,0.),
                zero,
                gparam
            )
    )
    inc = T.switch(
            T.gt(change,0.),
            - T.sgn(gparam) * new_delta,
            T.switch(
                T.lt(change,0.),
                zero,
                - T.sgn(gparam) * new_delta
            )
    )

    updates.append((previous_grad,new_previous_grad))
    updates.append((delta,new_delta))
    updates.append((previous_inc,inc))
    return param + inc * mask


def rprop(param,learning_rate,gparam,mask,updates,current_cost,previous_cost,
          eta_plus=1.01,eta_minus=0.1,max_delta=5, min_delta=1e-3):
    previous_grad = sharedX(numpy.ones(param.shape.eval()),borrow=True)
    delta = sharedX(learning_rate * numpy.ones(param.shape.eval()),borrow=True)
    previous_inc = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
    zero = T.zeros_like(param)
    one = T.ones_like(param)
    change = previous_grad * gparam

    new_delta = T.clip(
            T.switch(
                T.eq(gparam,0.),
                delta,
                T.switch(
                    T.gt(change,0.),
                    delta*eta_plus,
                    T.switch(
                        T.lt(change,0.),
                        delta*eta_minus,
                        delta
                    )
                )
            ),
            min_delta,
            max_delta
    )
    new_previous_grad = T.switch(
            T.eq(mask * gparam,0.),
            previous_grad,
            T.switch(
                T.gt(change,0.),
                gparam,
                T.switch(
                    T.lt(change,0.),
                    zero,
                    gparam
                )
            )
    )
    inc = T.switch(
            T.eq(mask * gparam,0.),
            zero,
            T.switch(
                T.gt(change,0.),
                - T.sgn(gparam) * new_delta,
                T.switch(
                    T.lt(change,0.),
                    zero,
                    - T.sgn(gparam) * new_delta
                )
            )
    )

    updates.append((previous_grad,new_previous_grad))
    updates.append((delta,new_delta))
    updates.append((previous_inc,inc))
    return param + inc * mask

def irprop(param,learning_rate,gparam,mask,updates,current_cost,previous_cost,
          eta_plus=1.5,eta_minus=0.25,max_delta=500, min_delta=1e-8):
    previous_grad = sharedX(numpy.ones(param.shape.eval()),borrow=True)
    delta = sharedX(learning_rate * numpy.ones(param.shape.eval()),borrow=True)
    previous_inc = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
    zero = T.zeros_like(param)
    one = T.ones_like(param)
    change = previous_grad * gparam

    new_delta = T.clip(
            T.switch(
                T.eq(mask * gparam,0.),
                delta,
                T.switch(
                    T.gt(change,0.),
                    delta*eta_plus,
                    T.switch(
                        T.lt(change,0.),
                        delta*eta_minus,
                        delta
                    )
                )
            ),
            min_delta,
            max_delta
    )
    new_previous_grad = T.switch(
            T.eq(mask * gparam,0.),
            previous_grad,
            T.switch(
                T.gt(change,0.),
                gparam,
                T.switch(
                    T.lt(change,0.),
                    zero,
                    gparam
                )
            )
    )
    inc = T.switch(
            T.eq(mask * gparam,0.),
            zero,
            T.switch(
                T.gt(change,0.),
                - T.sgn(gparam) * new_delta,
                T.switch(
                    T.lt(change,0.),
                    zero,
#                    - T.sgn(gparam) * new_delta
                    T.switch( 
                        T.gt(current_cost, previous_cost),
                        - T.sgn(gparam) * new_delta,
                        zero
                    )
                )
            )
    )

    updates.append((previous_grad,new_previous_grad))
    updates.append((delta,new_delta))
    updates.append((previous_inc,inc))
    return param + inc * mask

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh,dropout_rate=0,layerName='hidden'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.dropout_rate=dropout_rate
        self.layerName=layerName

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=layerName + '_W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=layerName + '_b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) * (1 - self.dropout_rate) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


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

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayers = []
        chain_n_in = n_in
        chain_in = input
        for (n_this,drop_this,name_this,activation_this) in n_hidden:
            l = HiddenLayer(rng=rng, input=chain_in, n_in=chain_n_in, n_out=n_this,
                    activation=activation_this,dropout_rate=drop_this,layerName=name_this)
            chain_n_in=n_this
            chain_in=l.output
            self.hiddenLayers.append(l)

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
             batch_size=20, n_hidden=(500,0), update_rule=sgd, n_in=28*28):
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
        dropout_rates[layer.layerName + '_W'] = layer.dropout_rate
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
                     update_rule=sgd,n_in=28*28):
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
        dropout_rates[layer.layerName + '_W'] = layer.dropout_rate
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
    update_rule=rprop
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
                  (2500,0.5,'h0',T.tanh),
                  (2000,0.5,'h1',T.tanh),
                  (1500,0.5,'h2',T.tanh),
                  (1000,0.5,'h2',T.tanh),
                  (500,0.5,'h3',T.tanh)
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
                  (2500,0.5,'h0',T.tanh),
                  (2000,0.5,'h1',T.tanh),
                  (1500,0.5,'h2',T.tanh),
                  (1000,0.5,'h2',T.tanh),
                  (500,0.5,'h3',T.tanh)
                 ]
        n_in = 784
    elif dataset_name == 'cifar10':
        learning_rate=0.001
        dataset='/local/cifar10/'
        pickled=False
        n_hidden=[
                  (3000,0.5,'h0',T.tanh),
                  (3000,0.5,'h1',T.tanh),
                  (3000,0.5,'h2',T.tanh),
                  (3000,0.5,'h3',T.tanh),
                  (3000,0.5,'h4',T.tanh)
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
        def update_rule(param,learning_rate,gparam,mask,updates,current_cost,previous_cost):
            return rprop(param,learning_rate,gparam,mask,updates,current_cost,previous_cost,
                         eta_plus=eta_plus,eta_minus=eta_minus,max_delta=max_delta,min_delta=min_delta)
        mlp=test_mlp(datasets,learning_rate, L1_reg, L2_reg, n_epochs,
            batch_size, n_hidden, update_rule = old_rprop, n_in = n_in)
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
