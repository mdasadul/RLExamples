from __future__ import print_function
import numpy as np 
import tensorflow as tf 

import gym

class Actor(object):
    def __init__(self, n_feature, n_action,sess,learning_rate = 0.01):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [1,n_feature],name='state')
        self.action = tf.placeholder(tf.int32,None, name='action')
        self.td_error = tf.placeholder(tf.float32, None, name='td_error')


        with tf.variable_scope('actor'):
            l1 = tf.layers.dense(self.state,
                                20,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.random_normal_initializer(0.,0.1),
                                bias_initializer=tf.constant_initializer(0.1),
                                name='l1')
            self.ac_prob = tf.layers.dense( l1,
                                        n_action,
                                        activation=tf.nn.softmax,
                                        kernel_initializer=tf.random_normal_initializer(0.,0.1),
                                        bias_initializer = tf.constant_initializer(0.1),
                                        name = 'activation_prob')
        
        with tf.variable_scope('td_error'):
            log_prob = tf.log(self.ac_prob[0,self.action])
            self.exp_v = tf.reduce_mean(self.td_error*log_prob)
        
        with tf.variable_scope('training_op'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v)

    def choose_action(self, state):
        #state = state[np.newaxis,:]
        ac_prob = self.sess.run(self.ac_prob, feed_dict = {self.state:state})
        return np.random.choice(np.arange(ac_prob.shape[1]), p=ac_prob.ravel())
    


    def learn(self, state, action,error):
        #state = state[np.newaxis,:]
        feeddict = {self.state: state, self.action:action,self.td_error:error}

        exp_v,_ = self.sess.run([self.exp_v,self.train_op], feed_dict = feeddict)
        return exp_v


class Critic(object):
    def __init__(self, n_feature, sess, learning_rate, gamma = 0.9):
        self.sess = sess
        a=np.random.randint(10,size=(10,20))

        self.state = tf.placeholder(tf.float32, [1,n_feature], name='state')
        self.reward = tf.placeholder(tf.float32, None, name = 'reward')
        self.v_ = tf.placeholder(tf.float32, [1,1], name = 'v')

        with tf.variable_scope('critics'):
            l1 = tf.layers.dense(self.state,
                                20,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.random_normal_initializer(0.,0.1),
                                bias_initializer=tf.constant_initializer(0.1),
                                name='l1')
            self.v = tf.layers.dense( l1,
                                        1,
                                        activation=None,
                                        kernel_initializer=tf.random_normal_initializer(0.,0.1),                                        bias_initializer = tf.constant_initializer(0.1),
                                        name = 'activation_prob')

            with tf.variable_scope('tf_error'):
                self.td_error = self.reward + gamma * self.v_ - self.v
                self.loss = tf.square(self.td_error)
            with tf.variable_scope('train_op'):
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            
    def learn(self, state, reward,state_):
        #state,state_ = state[np.newaxis,:],state_[np.newaxis,:]

        exp_v = self.sess.run(self.v, feed_dict={self.state:state_})

        feeddict = {self.reward: reward, self.v_:exp_v, self.state: state}
        _, error=self.sess.run([self.train_op,self.td_error],feeddict)

        return error

def step(source, predict):
    diff = np.array(source[0][1:])-np.array(predict[0][:-1])
    #print(diff)
    reward = 0-np.sum(np.square(diff))
    done = False
    if reward == 0:
        done = True
    return predict, reward, done

if __name__ =='__main__':
    
    n_feature = 9
    n_action = 10
    sess = tf.Session()
    input = np.random.randint(10,size=(10,10))
    print(input)
    actor = Actor(n_feature, n_action,sess,learning_rate= 0.001)
    critic = Critic(n_feature,sess,learning_rate=0.01)
    
    sess.run(tf.global_variables_initializer())

    num_tests = 1

    reward_list =[]
    jlist = []
    for t in range(num_tests):
        state = input[0:1,:-1]
        
        rewards = 0
        iter  = 0
        while True:
            iter +=1
            action = actor.choose_action(state)
            print('Action = ',action)
            #print(state)
            state_, reward, done=  step(state, input[action:action+1,1:] )
            #print(state_, reward)
            td_error = critic.learn(state,reward,state_)
            actor.learn(state, action, td_error)
            state = state_

            rewards += reward
            if done or iter<100:
                print('Total rewards: %d' %(rewards))
                break
        
        reward_list.append(rewards)
        #jlist.append(j)
    
    print(np.sum(reward_list)/num_tests)




