"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from deep_q_network import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001,gamma=0.9)



total_steps = 0
for i_episode in range(10):
    
    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # the higher the better
        # reward = abs(position - (-0.3))     # r in [0, 1]

        RL.save_experience(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward

        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2),
                  '| Steps', total_steps)
            break

        # if total_steps> 10000:
        #     break

        observation = observation_
        total_steps += 1

RL.plot_loss()
