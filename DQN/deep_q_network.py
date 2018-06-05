from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

class DeepQNetwork(object):
    def __init__(self, n_actions,
                n_features,
                learning_rate,
                gamma,
                epsilon_max=0.9,
                replace_target_step = 300,
                reply_memory_size = 5000,
                num_layers1 = 10,
                batch_size = 32,
                epsilon_increment=None,
                save_graph = False
                ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.replace_target_step = replace_target_step
        self.global_steps = 0
        self.reply_memory_size = reply_memory_size
        self.num_layers1 = num_layers1
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        
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

        self.total_cost=[]

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

        memory_index = self.memory_counter % self.reply_memory_size
        self.reply_memory[memory_index,:] = transition
        self.memory_counter +=1
    

    def choose_action(self, state):

        state = state[np.newaxis,:]
        if np.random.uniform()< self.epsilon:
            action_value = self.sess.run(self.eval_net,feed_dict={self.state:state})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(self.n_actions)
        return action

    
    def learn(self):
        
        if self.global_steps % self.replace_target_step == 0:
            self.sess.run(self.exchange_params_op)
            print('Eval --> Target')

        if self.memory_counter > self.reply_memory_size:
            sample_batch = np.random.choice(self.reply_memory_size,self.batch_size)
        else:
            sample_batch = np.random.choice(self.memory_counter,self.batch_size)
        
        batch_memory = self.reply_memory[sample_batch,:]
        
        next_state, q_eval = self.sess.run([self.next_states,self.eval_net], feed_dict={
            self.state_ : batch_memory[:,-self.n_features:],
            self.state : batch_memory[:,:self.n_features]
        })
        
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        eval_index = batch_memory[:,self.n_features].astype(int)
        reward = batch_memory[:,self.n_features+1]
        
        q_target[batch_index,eval_index] = reward + self.gamma*np.max(next_state,axis=1)

        loss,_ = self.sess.run([self.loss, self.train_op],feed_dict={self.state:batch_memory[:,:self.n_features], self.target_q:q_target})
        self.total_cost.append(loss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon <self.epsilon_max else self.epsilon_max

        self.global_steps +=1 

    def plot_loss(self):
        plt.plot(np.arange(len(self.total_cost)),self.total_cost)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()

