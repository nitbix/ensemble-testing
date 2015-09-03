#!/usr/bin/python

import gc
import sys
import numpy as np
import numpy.random
import theano
import theano.tensor as T
import yaml

import mlp
import parameters
from logistic_sgd import LogisticRegression
from data import Resampler, Transformer, sharedX, load_data, make_pretraining_set

class AveragingRunner:
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,members,x,y):
        self.members=members
        self.x = x
        self.y = y
        self.p_y_given_x = 0.
        self.p_y_given_x = sum([m.p_y_given_x for m in self.members]) / len(members)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.mean(T.neq(self.y_pred, y))


class MajorityVotingRunner:
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,members,x,y):
        self.members=members
        self.x = x
        self.y = y
        self.p_y_given_x = 0.
        self.p_y_given_x = sum([T.eq(T.max(m.p_y_given_x),m.p_y_given_x)
            for m in self.members])
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.mean(T.neq(self.y_pred, y))


class StackingRunner:
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,members,x,y,n_hidden,update_rule,n_epochs,batch_size,train_set,valid_set):
        self.members=members
        train_set_x,train_set_y = train_set
        valid_set_x,valid_set_y = valid_set
        self.train_input_x = theano.function(inputs=[],
                outputs=T.concatenate([m.p_y_given_x
                    for m in self.members],axis=1),
                givens={x:train_set_x})
        self.valid_input_x = theano.function(inputs=[],
                outputs=T.concatenate([m.p_y_given_x
                    for m in self.members],axis=1),
                givens={x:valid_set_x})
        print 'training stack head'
        self.head_x = T.concatenate([m.p_y_given_x
            for m in self.members],axis=1)
        self.stack_head = mlp.train_and_select(self.head_x,y,
                (sharedX(self.train_input_x(),borrow=True),train_set_y),
                (sharedX(self.valid_input_x(),borrow=True),valid_set_y),
                L1_reg=0.,L2_reg=0.,n_epochs=n_epochs,batch_size=batch_size,
                n_hidden=n_hidden,
                update_rule=update_rule,
                n_in=10*len(members))
        self.y_pred = self.stack_head.y_pred
        self.errors = self.stack_head.errors(y)


class EnsembleMethod(yaml.YAMLObject):

    def create(self,members,x,y,train_set,valid_set):
        raise NotImplementedException()


class Bagging(EnsembleMethod):
    """
    Create a Bagging Runner from parameters
    """

    yaml_tag = u'!Bagging'
    def __init__(self,voting=False):
        self.voting = voting

    def create(self,members,x,y,train_set,valid_set):
        if self.voting:
            return MajorityVotingRunner(members,x,y)
        else:
            return AveragingRunner(members,x,y)


class Stacking(EnsembleMethod):
    """
    Create a Bagging Runner from parameters
    """

    yaml_tag = u'!Stacking'
    def __init__(self,n_hidden,update_rule,n_epochs,batch_size):
        self.n_hidden = n_hidden
        self.update_rule = update_rule
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def create(self,members,x,y,dataset):
        return StackingRunner(members,x,y,self.n_hidden,self.update_rule,
                self.n_epochs,batch_size,train_set,valid_set)


if __name__ == '__main__':
    params = parameters.load_parameters(sys.argv[1])
    dataset = load_data(params.dataset,
                              shared = False,
                              pickled = params.pickled)
    pretraining_set = make_pretraining_set(dataset,params.pretraining)
    resampler = Resampler(dataset)
    train_set = resampler.get_train()
    valid_set = resampler.get_valid()
    mlp_training_dataset = (train_set,valid_set)
    x = T.matrix('x')
    y = T.ivector('y')
    members = []
    for i in range(0,params.ensemble_size):
        print 'training member {0}'.format(i)
        m = mlp.test_mlp(mlp_training_dataset, params, pretraining_set = pretraining_set)
        members.append(m)
        gc.collect()
    ensemble = params.method.create(members,x,y,train_set,valid_set)
    test_set_x, test_set_y = resampler.get_test()
    test_model = theano.function(inputs=[],
        on_unused_input='warn',
        outputs=ensemble.errors,
        givens={x:test_set_x, y:test_set_y})
    test_score = test_model()
    print 'Final error: {0} %'.format(test_score * 100.)
