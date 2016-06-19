#!/usr/bin/python

import gc
import sys
import numpy as np
import numpy.random
import argparse
import dill
from pymongo import MongoClient
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a single MLP')
    parser.add_argument('params_file', help='the parameters file')
    parser.add_argument('save_file', nargs='?',
                        help='the file where the trained MLP is to be saved')
    parser.add_argument('--seed', type=int, nargs='?',
                        help='random seed to use for this sim')
    parser.add_argument('--results-db', nargs='?',
                        help='mongodb db name for storing results')
    parser.add_argument('--results-host', nargs='?',
                        help='mongodb host name for storing results')
    parser.add_argument('--results-table', nargs='?',
                        help='mongodb table name for storing results')
    parser.add_argument('--device', nargs='?',
                        help='gpu/cpu device to use for training')

    args = parser.parse_args()
    #this needs to come before all the toupee and theano imports
    #because theano starts up with gpu0 and never lets you change it
    if args.device is not None:
        if 'THEANO_FLAGS' in os.environ is not None:
            env = os.environ['THEANO_FLAGS']
            env = re.sub(r'/device=[a-zA-Z0-9]+/',r'/device=' + args.device, env)
        else:
            env = 'device=' + args.device
        os.environ['THEANO_FLAGS'] = env

    arg_param_pairings = [
        (args.seed, 'random_seed'),
        (args.results_db, 'results_db'),
        (args.results_host, 'results_host'),
        (args.results_table, 'results_table'),
    ]
    from toupee import config
    params = config.load_parameters(args.params_file)

    def arg_params(arg_value,param):
        if arg_value is not None:
            params.__dict__[param] = arg_value

    for arg, param in arg_param_pairings:
        arg_params(arg,param)

    from toupee.data import *
    from toupee.mlp import MLP, test_mlp
    from toupee import common
    import theano
    import theano.tensor as T
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
    if args.save_file is not None:
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
    print 'Final Ensemble error: {0} %'.format(test_score * 100.)

    if 'results_db' in params.__dict__:
        if 'results_host' in params.__dict__:
            host = params.results_host
        else:
            host = None
        print "saving results to {0}@{1}".format(params.results_db,host)
        conn = MongoClient(host=host)
        db = conn[params.results_db]
        if 'results_table' in params.__dict__: 
            table_name = params.results_table
        else:
            table_name = 'results'
        table = db[table_name]
        results = {
                    "params": params.__dict__,
                    "test_losses" : test_losses,
                    "test_score" : test_score,
                  }
        table.insert(json.loads(json.dumps(results,default=common.serialize)))
    
    train,valid,test = dataset
    train_x,train_y = train
    valid_x,valid_y = valid
    test_x,test_y = test
    shared_train_x = sharedX(train_x)
    shared_valid_x = sharedX(valid_x)
    shared_test_x = sharedX(test_x)
    distilled_train_y = ensemble.classify(shared_train_x).eval()
    distilled_set = (
                        (shared_train_x,
                            sharedX(distilled_train_y, dtype = numpy.int32)),
                        (shared_valid_x,
                            sharedX(valid_y, dtype = numpy.int32)),
                        (shared_test_x,
                            sharedX(test_y, dtype = numpy.int32)),
                    )
    #y = T.vector()
    mlp = test_mlp(distilled_set, params, y = y)
