import yaml 
import activations
import update_rules

class Parameters(object):

#	def __init__(self):
#        self.dataset = 'mnist'
#        self.L1_reg = 0
#        self.L2_reg = 0
#        self.n_epochs = 100
#        self.transform = False
#        self.learning_rate = 0.001
#        self.pretraining = None
#        self.pretraining_passes = 1
#        self.training_method = 'normal'
#        self.update_rule = update_rules.sgd
#        self.batch_size = 300
#        self.n_hidden = [
#                  ('flat',(2500,0.5,'h0',T.tanh)),
#                  ('flat',(2000,0.5,'h1',T.tanh)),
#                  ('flat',(1500,0.5,'h2',T.tanh)),
#                  ('flat',(1000,0.5,'h2',T.tanh)),
#                  ('flat',(500,0.5,'h3',T.tanh))
#                 ]
#        n_in = 784

    def __init__(self, **entries): 
        self.__dict__.update(entries)

def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    return Parameters(**r)
