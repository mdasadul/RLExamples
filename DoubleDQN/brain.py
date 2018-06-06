import numpy as np 
import tensorflow as tf 

class DoubleDQN(object):
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate = 0.01,
        gamma = 0.9,
        max_epsilon = 0.9,
        replace_target_step = 300,
        double_dqn = False,
        sess = None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.replace_target_step = replace_target_step
        self.double_dqn = double_dqn
        self.sess = sess 

        if self.sess ==None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        
        self.q = None
    

    def create_network(self):
        pass
    
    def choose_action(self, observation):
        pass
    
    def learn(self):
        pass
