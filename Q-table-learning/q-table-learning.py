import gym  
import numpy as np 

environment = gym.make('FrozenLake-v0')

Q = np.zeros((environment.observation_space.n, environment.action_space.n))
lr = 0.8
y = 0.95
num_episodes = 1000

# create lists to contain total rewards and steps per episode
total_reward = []
for i in range(num_episodes):
    state = environment.reset()
    all_reward = 0
    d = False
    j = 0

    while j < 100:
        j +=1
        action = np.argmax(Q[state,:]+ np.random.randn(1,environment.action_space.n)*(1./(i+1))) 
        new_state, reward, done, _ = environment.step(action)
        Q[state,action] = Q[state, action] + lr*(reward+ y*np.max(Q[new_state,:])-Q[state,action])
        all_reward +=reward

        state = new_state
        if done == True:
            break
    total_reward.append(all_reward)

print("Score over time: "+ str(sum(total_reward)/num_episodes))

print("Final Q-table Values")
print(Q)

# Lets play FrozenLake by using the Q-table

#environment.reset()
for episode in range(1):
    state = environment.reset()
    step = 0
    print("Episode", episode)

    for step in range(100):
        environment.render()
        action = np.argmax(Q[state,:])

        new_state, reward, done, info = environment.step(action)
        print(info,new_state)
        if done:
            break
        state = new_state
