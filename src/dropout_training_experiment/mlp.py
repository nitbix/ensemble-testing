#!/usr/bin/python
"""
Representation of a Multi-Layer Perceptron

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import dill
import argparse
import os
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a single MLP')
    parser.add_argument('params_file', help='the parameters file')
    parser.add_argument('save_file', nargs='?',
                        help='the file where the trained MLP is to be saved')
    parser.add_argument('--seed', type=int, nargs='?',
                        help='random seed to use for this sim')
    parser.add_argument('--epochs', type=int, nargs='?',
                        help='number of epochs to run')
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
        (args.epochs, 'n_epochs'),
    ]
    from toupee import config
    params = config.load_parameters(args.params_file)

    def arg_params(arg_value,param):
        if arg_value is not None:
            params.__dict__[param] = arg_value

    for arg, param in arg_param_pairings:
        arg_params(arg,param)

    from toupee import data
    from toupee.mlp import MLP, test_mlp
    dataset = data.load_data(params.dataset,
                             resize_to = params.resize_data_to,
                             shared = False,
                             pickled = params.pickled,
                             center_and_normalise = params.center_and_normalise,
                             join_train_and_valid = params.join_train_and_valid)
    pretraining_set = data.make_pretraining_set(dataset,params.pretraining)
    mlp = test_mlp(dataset, params, pretraining_set = pretraining_set)
    if args.save_file is not None:
        dill.dump(mlp,open(args.save_file,"wb"))
