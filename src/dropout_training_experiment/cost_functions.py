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
import yaml

class CostFunction(yaml.YAMLObject):

    def __call__(model,y):
        raise NotImplementedError()

class CrossEntropy(CostFunction):
    yaml_tag = u'!CrossEntropy'
    def __call__(model, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
        return T.nnet.binary_crossentropy(y,model.y)

class NegLogLikelihood(CostFunction):
    yaml_tag = u'!NegLogLikelihood'
    def __call__(model, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
        return -T.mean(T.log(model.p_y_given_x)[T.arange(y.shape[0]), y])

class MSE(CostFunction):
    yaml_tag = u'!MSE'
    def __call__(model,y):
        """
        Return the mean squared error
        """

        if y.ndim != model.y.ndim:
            raise TypeError('y should have the same shape as model.y',
                ('y', y.type, 'y_pred', model.y.type, 'layer', model.layer_name))
        return T.mean((model.y - y) ** 2)

