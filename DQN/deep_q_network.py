from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

class DeepQNetwork(object):
    def __init__(self, n_actions,
                n_features,
                learning_rate,
                gamma,
                epsilon=0.9,
                bum_replace_targets = 300,
                save_graph = False):
        pass
    
    def __build_graph(self):
        pass
    
    def save_experience(self, state, action, reward, state_):
        pass
    
    def learn(self):
        pass
    


