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
loss = tf.reduce_sum(tf.square(next_move-possible_move))

optimizer = tf.train.AdamOptimizer(0.01)
step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

gamma = 0.99
epsilon = 0.1
num_episodes = 3000
list_reward = []

with tf.Session() as sess:
    sess.run(init)
    for episode in range(num_episodes):
        state = environment.reset()
        rewards = 0
        num_tries = 100
        for t in range(num_tries):
            feed_dict = np.identity(16)[state:state+1]
            states, action = sess.run([possible_move, predict], feed_dict={input_state:feed_dict})
            if np.random.rand(1) < epsilon:
                action[0] = environment.action_space.sample()
            
            new_state, reward, done, info = environment.step(action[0])
                
            feed_dict = np.identity(16)[new_state:new_state+1]
            moves = sess.run(possible_move, feed_dict={input_state:feed_dict})
            max_moves = np.max(moves)

            target_moves = states
            target_moves[0,action[0]] = reward + gamma * max_moves

            feed_dict = np.identity(16)[state:state+1]
            _, Weight = sess.run([step,weights], feed_dict={input_state:feed_dict,next_move:target_moves})

            rewards += reward
            state = new_state
            if done == True:
                epsilon = 1./((t/50) + 10)
                break   
                
        list_reward.append(rewards)

print("percent of successful episode: "+ str(sum(list_reward)/num_episodes))
    
plt.plot(list_reward)
plt.show()


