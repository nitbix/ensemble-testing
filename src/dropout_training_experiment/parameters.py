import yaml 
import activations
import update_rules

class Parameters(object):

    def __init__(self, **entries): 
        self.__dict__.update(entries)

def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    return Parameters(**r)
