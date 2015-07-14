#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from data import sharedX


class Layer:
    def __init__(self,rng,inputs,n_in,n_out,activation,
                 dropout_rate,layer_name,W=None,b=None):
        self.inputs = inputs
        self.dropout_rate=dropout_rate
        self.layer_name=layer_name

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name=layer_name + '_W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=layer_name + '_b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]


class FlatLayer(Layer):
    def __init__(self, rng, inputs, n_in, n_out, W=None, b=None,
                 activation=T.tanh,dropout_rate=0,layer_name='hidden'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The default nonlinearity used here is tanh

        Hidden unit activation is given by: a(dot(input,W) + b)

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
        Layer.__init__(self,rng,inputs,n_in,n_out,activation,dropout_rate,layer_name,W,b)
         
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
        lin_output = T.dot(self.inputs, self.W) * (1 - self.dropout_rate) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model

class ConvolutionalLayer(Layer):
    """
    A Convolutional Layer, as per Convolutional Neural Networks. Includes filter, and pooling.
    """
    def __init__(self, rng, inputs, input_shape, filter_shape, pool_size, W=None, b=None,
             activation=T.tanh,dropout_rate=0,layer_name='conv',border_mode='valid'):
        """
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
        assert input_shape[1] == filter_shape[1]

        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.border_mode = border_mode
        self.fan_in = numpy.prod(self.filter_shape[1:])
        self.fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(pool_size)

        #W and b are slightly different
        if W is None:
                W_bound = numpy.sqrt(6. / (self.fan_in + self.fan_out))
                initial_W = numpy.asarray( rng.uniform(
                                       low=-W_bound, high=W_bound,
                                       size=filter_shape),
                                       dtype=theano.config.floatX)

                if activation == T.nnet.sigmoid:
                    initial_W *= 4
                W = theano.shared(value = initial_W, name = 'W')
        if b is None:
                b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
                b = theano.shared(value=b_values, name='b')

        Layer.__init__(self,rng,T.reshape(inputs,input_shape,ndim=4),filter_shape[0],
                filter_shape[1], activation,dropout_rate,layer_name,W,b)
        self.delta_W = sharedX(
            value=numpy.zeros(filter_shape),
            name='{0}_delta_W'.format(self.layer_name))
        self.delta_b = sharedX(
            value=numpy.zeros_like(self.b.get_value(borrow=True)),
            name='{0}_delta_b'.format(self.layer_name))
        self.conv_out = conv.conv2d(
            input=self.inputs,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=input_shape,
            border_mode=self.border_mode)
        self.y_out = activation(self.conv_out + self.b.dimshuffle('x',0,'x','x'))
        self.pooled_out = downsample.max_pool_2d(input=self.y_out,ds=self.pool_size,ignore_border=True)
        self.output = self.pooled_out
