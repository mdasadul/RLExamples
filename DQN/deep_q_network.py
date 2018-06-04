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
        self.memory_counter = 0
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
        
        self.target_q = tf.placeholder(tf.float32, shape=[None, self.n_actions],name='target_q')
        weight_initializer, bias_initializer =  tf.random_normal_initializer(0.0,0.4),tf.constant_initializer(0.1)
        with tf.variable_scope('eval_net'):
            collection_names= ['eval_params',tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1',shape=[self.n_features,self.num_layers1],
                initializer=weight_initializer, collections=collection_names)

                b1 = tf.get_variable('b1',shape=[1,self.num_layers1], initializer= bias_initializer, collections= collection_names)
                layer1 = tf.nn.relu(tf.matmul(self.state,w1)+b1)
            
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', shape=[self.num_layers1,self.n_actions], 
                initializer=weight_initializer, collections=collection_names)

                b2 = tf.get_variable('b2',shape=[1,self.n_actions],initializer=bias_initializer,collections=collection_names)

                self.eval_net = tf.matmul(layer1,w2)+b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.eval_net,self.target_q))
        with tf.variable_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # build target network

        self.state_ = tf.placeholder(tf.float32, shape=[None,self.n_features], name='next_state')
        
        with tf.variable_scope('target_net'):
            collection_names = ['target_params',tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1',shape=[self.n_features,self.num_layers1],
                initializer=weight_initializer, collections=collection_names)

                b1 = tf.get_variable('b1',shape=[1,self.num_layers1], initializer= bias_initializer, collections= collection_names)
                layer1 = tf.nn.relu(tf.matmul(self.state_,w1)+b1)
            
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', shape=[self.num_layers1,self.n_actions], 
                initializer=weight_initializer, collections=collection_names)

                b2 = tf.get_variable('b2',shape=[1,self.n_actions],initializer=bias_initializer,collections=collection_names)

                self.next_states = tf.matmul(layer1,w2)+b2




    
    def save_experience(self, state, action, reward, state_):
        
        transition = np.hstack((state,[action,reward],state_))

        memory_index =  self.reply_memory_size % self.memory_counter
        self.reply_memory[memory_index,:] = transition
        self.memory_counter +=1
    

    def choose_action(self, state):
        pass


    
    def learn(self):
        pass
    


