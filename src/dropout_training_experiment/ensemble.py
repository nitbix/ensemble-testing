#!/usr/bin/python

import gc
import sys
import numpy as np
import numpy.random
import theano
import theano.tensor as T
import mlp
from logistic_sgd import LogisticRegression
from data import Resampler, Transformer, sharedX, load_data

class Averaging:
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,ensemble,x,y):
        self.ensemble=ensemble
        self.x = x
        self.y = y
        self.p_y_given_x = 0.
        self.p_y_given_x = sum([m.p_y_given_x for m in self.ensemble]) / len(ensemble)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.mean(T.neq(self.y_pred, y))


class MajorityVoting:
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,ensemble,x,y):
        self.ensemble=ensemble
        self.x = x
        self.y = y
        self.p_y_given_x = 0.
        self.p_y_given_x = sum([T.eq(T.max(m.p_y_given_x),m.p_y_given_x)
            for m in self.ensemble])
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.mean(T.neq(self.y_pred, y))


class Stacking:
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,x,y,ensemble,n_hidden,update_rule,n_epochs,batch_size,train_set,valid_set):
        self.ensemble=ensemble
        train_set_x,train_set_y = train_set
        valid_set_x,valid_set_y = valid_set
        self.train_input_x = theano.function(inputs=[],
                outputs=T.concatenate([m.p_y_given_x
                    for m in self.ensemble],axis=1),
                givens={x:train_set_x})
        self.valid_input_x = theano.function(inputs=[],
                outputs=T.concatenate([m.p_y_given_x
                    for m in self.ensemble],axis=1),
                givens={x:valid_set_x})
        print 'training stack head'
        self.head_x = T.concatenate([m.p_y_given_x
            for m in self.ensemble],axis=1)
        self.stack_head = mlp.train_and_select(self.head_x,y,
                (sharedX(self.train_input_x(),borrow=True),train_set_y),
                (sharedX(self.valid_input_x(),borrow=True),valid_set_y),
                L1_reg=0.,L2_reg=0.,n_epochs=n_epochs,batch_size=batch_size,
                n_hidden=n_hidden,
                update_rule=update_rule,
                n_in=10*len(ensemble))
        self.y_pred = self.stack_head.y_pred
        self.errors = self.stack_head.errors(y)

if __name__ == '__main__':

    learning_rate=0.01
    L1_reg=0.00
    L2_reg=0.00
    n_epochs=50
    dataset='mnist.pkl.gz'
    batch_size=300
    resample_size=50000
    n_hidden=[(2500,0.5,'h0',T.tanh),
              (2000,0.5,'h1',T.tanh),
              (1500,0.5,'h2',T.tanh),
              (1000,0.5,'h2',T.tanh),
              (500,0.5,'h3',T.tanh)
             ]
    method = 'bagging'
    ensemble_size = 10
    eta_plus = 1.01
    eta_minus = 0.1
    max_delta = 5
    min_delta = 1e-3
    for arg in sys.argv[1:]:
        if arg[0]=='-':
            exec(arg[1:])
    dataset = load_data(dataset,shared=False)
    def transform_dataset(dataset):
        train,valid,test = dataset
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test
        aggregate_x = np.concatenate((train_x, valid_x), axis=0)
        aggregate_y = np.concatenate((train_y, valid_y), axis=0)
        t = Transformer((aggregate_x,aggregate_y),28,28)
        aggregate_train = t.get_data()
        aggregate_valid = (aggregate_x, aggregate_y)
        return (aggregate_train,aggregate_valid,test)
    #dataset = transform_dataset(dataset)
    def update_rule(param,learning_rate,gparam,mask,updates,current_cost,previous_cost):
        return mlp.rprop(param,learning_rate,gparam,mask,updates,current_cost,previous_cost,eta_plus=eta_plus,eta_minus=eta_minus,max_delta=max_delta,min_delta=min_delta)
    resampler = Resampler(dataset)
    x = T.matrix('x')
    y = T.ivector('y')
    members = []
    for i in range(0,ensemble_size):
        print 'training member {0}'.format(i)
        m=mlp.train_and_select(x,y,resampler.make_new_train(resample_size),
                resampler.get_valid(),learning_rate, L1_reg, L2_reg, n_epochs,
                batch_size, n_hidden, update_rule = update_rule)
        members.append(m)
        gc.collect()
    if method ==  'bagging':
        mv = Averaging(members,x,y)
    elif method == 'stacking':
        mv = Stacking(x,y,members,[
                #(ensemble_size * 500,0,'s0',T.tanh),
                #(ensemble_size * 400,0,'s1',T.tanh),
               # (ensemble_size * 300,0,'s2',T.tanh),
                (ensemble_size * 200,0,'s3',T.tanh),
                (ensemble_size * 100,0,'s4',T.tanh),
            ],
            update_rule=mlp.rprop,
            n_epochs=100,
            batch_size=batch_size,
            train_set=resampler.get_train(),
            valid_set=resampler.get_valid())
    test_set_x, test_set_y = resampler.get_test()
    test_model = theano.function(inputs=[],
        on_unused_input='warn',
        outputs=mv.errors,
        givens={x:test_set_x, y:test_set_y})
    test_score = test_model()
    print 'Final error: {0} %'.format(test_score * 100.)
