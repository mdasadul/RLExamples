import numpy as np 
import tensorflow as tf 

class DoubleDQN(object):
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate = 0.005,
        gamma = 0.9,
        max_epsilon = 0.9,
        replace_target_step = 300,
        double_dqn = False,
        sess = None,
        memory_size = 3000,
        hidden_layer_size = 20,
        batch_size = 32,
        epsilon_increment = 0.01
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
        self.batch_size = batch_size
        self.memory = np.zeros((memory_size, n_features * 2 + 2))
        self.epsilon_increment = epsilon_increment
        self.epsilon_max = max_epsilon
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        

        self.build_graph()
        eval_params = tf.get_collection('eval_net_params')
        target_params = tf.get_collection('target_net_params')
        self.replace_target_op = [tf.assign(t,e) for (e,t) in zip(eval_params, target_params)]


        if self.sess ==None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        
        self.q = []
        self.q_running = 0

        
        self.memory_counter = 0
        self.global_step = 0
        self.total_loss =[]

    def build_graph(self):
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
        c_name_target = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope('eval_net'):
            self.eval_net = build_net(c_name_eval,self.state)

        self.q_target = tf.placeholder(dtype = tf.float32,
                                        shape= [None,self.n_actions],
                                        name='q_target')
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.eval_net))
        with tf.variable_scope('train'):
            self.training_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.state_ = tf.placeholder(dtype = tf.float32,
                                    shape=[None,self.n_features],
                                    name='next_state')
        with tf.variable_scope('target_net'):
            self.q_next = build_net(c_name_target,self.state_)



    def save_transition(self, state, action,reward,state_):
        transition = np.hstack((state,[action,reward], state_))
        index = self.memory_counter % self.memory_size
        self.memory[index:,] = transition
        self.memory_counter += 1
        
    def choose_action(self, observation):
        observation = observation[np.newaxis,:]
        eval_actions = self.sess.run(self.eval_net, feed_dict={self.state:observation})
        action = np.argmax(eval_actions)
        self.q_running = self.q_running* 0.99 + 0.01* eval_actions.max()
        self.q.append(self.q_running)
        if self.epsilon < np.random.uniform():
            action = np.random.randint(0,self.n_actions)

        return action

    
    def learn(self):
        if self.global_step % self.replace_target_step ==0:
            self.sess.run(self.replace_target_op)
        
        if self.memory_counter > self.memory_size:
            index = np.random.choice(self.memory_size, self.batch_size )
        else:
            index = np.random.choice(self.memory_counter, self.batch_size )
        
        batch_memory = self.memory[index,:]


        q_next, q_eval_next = self.sess.run([self.q_next,self.eval_net],feed_dict={self.state: batch_memory[:,-self.n_features:],
                                    self.state_: batch_memory[:,-self.n_features:] })
        
        q_eval = self.sess.run(self.eval_net, feed_dict={self.state: batch_memory[:,:self.n_features]})
        
        q_index = np.arange(self.batch_size)
        eval_index = batch_memory[:,self.n_features].astype(int)
        reward = batch_memory[:,self.n_features+1]

        if self.double_dqn:
            eval_index_q = np.argmax(q_eval_next,axis = 1)
            selected_next_q = q_next[q_index,eval_index_q]
        else:
            selected_next_q = np.max(q_next, axis= 1)

        q_target = q_eval.copy()
        q_target[q_index, eval_index] = reward + self.gamma * selected_next_q

        _, loss = self.sess.run([self.loss,self.training_op],feed_dict ={
                    self.q_target:q_target, 
                    self.state: batch_memory[:,:self.n_features]
        })

        self.total_loss.append(loss)
        
        
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon <self.epsilon_max else self.epsilon_max
        self.global_step +=1



        

