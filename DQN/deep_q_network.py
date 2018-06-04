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
                replace_target_step = 300,
                reply_memory_size = 5000,
                num_layers1 = 10,
                num_layers2 = 20,
                save_graph = False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.replace_target_step = replace_target_step
        self.global_steps = 0
        self.reply_memory_size = reply_memory_size
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        # each state has two information and there are two states+action+reward=6
        self.reply_memory = np.zeros((self.reply_memory_size,n_features*2+2))
        self.__build_graph()

        target_params = tf.get_collection('target_params')
        eval_params = tf.get_collection('eval_params')
        self.exchange_params_op = [tf.assign(e,t) for (e,t)in zip(eval_params, target_params)]

        self.sess = tf.Session()

        if save_graph:
            tf.summary.FileWriter('logs',self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())

    def __build_graph(self):
        self.state = tf.placeholder(tf.float32, shape=[None,self.n_features], name='present_state')
        

        with tf.variable_scope('eval_net'):
            collection_names, weight_initializer, bias_initializer = ['eval_params',tf.GraphKeys.GLOBAL_VARIABLES], tf.random_normal_initializer(0.0,0.4),
            tf.constant_initializer(0.1)
            


    
    def save_experience(self, state, action, reward, state_):
        pass
    
    def learn(self):
        pass
    


