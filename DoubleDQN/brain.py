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
        memory_size = 5000,
        hidden_layer_size = 10
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.replace_target_step = replace_target_step
        self.double_dqn = double_dqn
        self.sess = sess 
        self.memory_size = memory_size
        self.hidden_layer_size = hidden_layer_size

        self.memory = np.zeros((memory_size, n_features * 2 + 2))
        self.create_network()

        eval_params = tf.get_collection('eval_net_params')
        target_params = tf.get_collection('target_net_params')

        self.replace_target_op = [tf.assign(t,e) for (e,t) in zip(eval_params, target_params)]


        if self.sess ==None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        
        self.q = None

        self.epsilon = 0

    def create_network(self):
        self.state = tf.placeholder(dtype = tf.float32,
                                    shape=[None,self.n_features],
                                    name='current_state')

        weight_init = tf.random_normal_initializer(mean = 0.0, stddev = 0.3)
        bias_init = tf.constant_initializer(0.1)
        def build_net(c_name,state):

            with tf.variable_scope('layer1'):
                w1 = tf.get_variable(
                    'w1',
                    shape=[self.n_features, self.hidden_layer_size],
                    initializer= weight_init ,
                    collections=c_name)
                b1 = tf.get_variable(
                    'b1',
                    shape=[1,self.hidden_layer_size],
                    initializer= bias_init,
                    collections=c_name)
                
            layer_1 = tf.nn.relu(tf.matmul(state,w1)+b1)

            with tf.variable_scope('layer2'):
                w2 = tf.get_variable(
                    'w2',
                    shape=[self.hidden_layer_size,self.n_actions],
                    initializer=weight_init,
                    collections= c_name)
                b2 = tf.get_variable(
                    'b2',
                    shape=[1,self.n_actions],
                    initializer=bias_init,
                    collections=c_name)
            layer2 = tf.matmul(layer_1,w2)+b2

            return layer2 

        c_name_eval = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        c_name_target = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        
        self.eval_net = build_net(c_name_eval,self.state)

        self.q_target = tf.placeholder(dtype = tf.float32,
                                        shape= [None,self.n_actions],
                                        name='q_target')
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.eval_net))
        with tf.variable_scope('train'):
            self.training_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.state_ = tf.placeholder(dtype = tf.float32,
                                    shape=[None,self.n_features],
                                    name='next_state')
        self.target_net = build_net(c_name_target,self.state_)
        



    def choose_action(self, observation):
        pass
    
    def learn(self):
        pass
