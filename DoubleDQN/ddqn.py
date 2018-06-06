import numpy as np 
import tensorflow as tf 
import gym 
import matplotlib.pyplot as plt

from brain import DoubleDQN

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
ACTION_SPACE = 11


sess = tf.Session()
with tf.variable_scope('Normal_DQN'):
    normal_dqn = DoubleDQN(ACTION_SPACE,n_features=3,sess=sess)
with tf.variable_scope('DoubleDQN'):
    double_dqn = DoubleDQN(ACTION_SPACE,n_features=3, sess=sess, double_dqn=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    step = 0
        # initial observation
    observation = env.reset()

    while True:
        # fresh env
        if step  > 11000: 
            env.render() 

        # RL choose action based on observation
        action = RL.choose_action(observation)
        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)
        # RL take action and get next observation and reward
        observation_, reward, _, _= env.step(np.array([f_action]))

        reward /=10

        RL.save_transition(observation, action, reward, observation_)

        if (step > 3000):
            RL.learn()


        
        # break while loop when end of this episode
        if step > 15000:
            break
        # swap observation
        observation = observation_

        step += 1

    # end of game
    
    print('game over')
    #env.destroy()
    return RL.q


q_normal = train(normal_dqn)
q_double = train(double_dqn)

plt.plot(np.array(q_normal), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()
