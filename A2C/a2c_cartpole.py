import numpy as np 
import tensorflow as tf 
import gym 


env = gym.make('CartPole-v0')

class Actor(object):
    def __init__(self, n_features,n_action, sess,learning_rate=0.001):
        self.n_features = n_features
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.sess = sess

        self.state = tf.placeholder(tf.float32,shape=[None,self.n_features], name= 'currect_state')
        self.action = tf.placeholder(tf.int32,None,name='Action')
        self.error = tf.placeholder(tf.float32, shape=None,name = 'error')
        with tf.variable_scope('actor'):
            l1 = tf.layers.dense(inputs = self.state,
                                units = 20,
                                activation= tf.nn.relu,
                                kernel_initializer=tf.random_normal_initializer(0.,0.2),
                                bias_initializer=tf.constant_initializer(0.1),
                                name = 'l1' )
        
        
            self.action_prob = tf.layers.dense(inputs = l1,
                                units = self.n_action,
                                activation=tf.nn.softmax,
                                kernel_initializer= tf.random_normal_initializer(0.,0.2),
                                bias_initializer=tf.constant_initializer(0.1),
                                name= 'l2'
                                )
        
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.action_prob[0,self.action])
            self.exp_v = tf.reduce_mean(log_prob * self.error )
        with tf.variable_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.exp_v)

    
    def choose_action(self, observation):
        observation = observation[np.newaxis,:]
        action_prob  = self.sess.run(self.action_prob, feed_dict = {self.state:observation})

        action = np.random.choice(action_prob.shape[1],action_prob.ravel())
        return action 


    def learn(self, state, action, error):
        state = state[np.newaxis,:]
        feeddict = {self.state:state, self.action:action, self.error:error}
        _,exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict = feeddict)
        return exp_v

class Critic(object):
    def __init__(self, n_features, sess, learning_rate= 0.01,gamma = 0.9):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.sess = sess
        self.gamma = gamma

        self.state = tf.placeholder(tf.float32,shape=[None,self.n_features], name= 'currect_state')
        self.v_ = tf.placeholder(tf.float32,[1,1],name='v_')
        self.r = tf.placeholder(tf.float32, shape=None,name = 'error')
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(inputs = self.state,
                                units = 20,
                                activation= tf.nn.relu,
                                kernel_initializer=tf.random_normal_initializer(0.,0.2),
                                bias_initializer=tf.constant_initializer(0.1),
                                name = 'l1' )
        
        
            self.v = tf.layers.dense(inputs = l1,
                                units = 1,
                                activation=None,
                                kernel_initializer= tf.random_normal_initializer(0.,0.2),
                                bias_initializer=tf.constant_initializer(0.1),
                                name= 'l2'
                                )
        
        with tf.variable_scope('td_error'):
            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
    def learn(self, state, reward,state_):
        state,state_ = state[np.newaxis,:], state_[np.newaxis,:]
        
        v_ = self.sess.run([self.v],feed_dict={self.state: state_})
        feeddict = {self.state:state,self.v_:v_,self.r:reward}
        td_error, _ ,_=self.sess.run([self.td_error,self.loss,self.train_op], feed_dict = feeddict)

        return td_error

        

n_features = env.observation_space.shape[0]
n_action = env.action_space.n

session = tf.Session()

actor = Actor(n_features,n_action, session,)
critics = Critic(n_features, session)
session.run(tf.global_variables_initializer())

for episode in range(1000):
    state = env.reset()
    while True:
        #action = env.step(state)
        env.render()
        
        action = actor.choose_action(state)
        state_, reward, done, info = env.step(action)

        td_error = critics.learn(state, reward, state_)
        actor.learn(state, action, td_error)
        state = state_







