import gym  
import numpy as np 
import random
import tensorflow as tf 
import matplotlib.pyplot as plt 

environment = gym.make('FrozenLake-v0')

tf.reset_default_graph()

input_state = tf.placeholder(shape=[1,16],dtype=tf.float32)
weights = tf.Variable(tf.random_uniform([16,4],0,0.01))

possible_move = tf.matmul(input_state,weights)
predict = tf.argmax(possible_move,1)


next_move = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_move,possible_move))

optimizer = tf.train.AdamOptimizer(0.01)
step = optimizer.minimize(loss)

init = tf.initialize_all_variables()

gamma = 0.99
epsilon = 0.1
num_episodes = 2000
list_reward = []

with tf.Session() as sess:
    sess.run(init)
    for episode in range(num_episodes):
        state = environment.reset()
        reward = 0
        num_tries = 100
        for t in range(num_tries):
            feed_dict = np.identity(16)[state:state+1]
            moves, state = sess.run([possible_move, predict], feed_dict={input_state:feed_dict})
            
