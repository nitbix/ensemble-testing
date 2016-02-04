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

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams                                                                                                                    
from toupee import data
from toupee.data import Resampler, Transformer, sharedX
from toupee import update_rules
from toupee import layers
from toupee import config 
from toupee import cost_functions
from toupee.mlp import MLP, test_mlp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a single MLP')
    parser.add_argument('params_file', help='the parameters file')
    parser.add_argument('save_file', nargs='?',
                        help='the file where the trained MLP is to be saved')
    parser.add_argument('--seed', type=int, nargs='?',
                        help='random seed to use for this sim')
    args = parser.parse_args()
    params = config.load_parameters(args.params_file)
    if args.seed is not None:
        params.random_seed = args.seed
    dataset = data.load_data(params.dataset,
                             resize_to = params.resize_data_to,
                             shared = True,
                             pickled = params.pickled,
                             center_and_normalise = params.center_and_normalise)
    pretraining_set = data.make_pretraining_set(dataset,params.pretraining)
    mlp = test_mlp(dataset, params, pretraining_set = pretraining_set)
    if args.save_file is not None:
        dill.dump(mlp,open(args.save_file,"wb"))
