#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import yaml 
import activations
import update_rules
import ensemble_methods
import parameters

def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    return parameters.Parameters(**r)
