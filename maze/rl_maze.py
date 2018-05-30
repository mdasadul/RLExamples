from create_maze import Maze
import numpy as np 
import pandas as pd 


class RLMaze(object):
    def __init__(self, actions, alpha, gamma, epsilon, terminal):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.terminal = terminal
        self.q_table = pd.DataFrame(columns=actions)
        
    def choose_action(self, state):
        self.check_state_exists(state)
        if np.random.uniform() > self.epsilon :
            action_name = np.random.choice(self.actions)
        else:
            action_name = self.q_table.loc[state,:].max()
        return action_name


    def check_state_exists(self,state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = [0]*self.actions
            

    def reinforcement_learning(self,state,action,reward,new_state):
        self.check_state_exists(new_state)
        if new_state !=self.terminal:
            q_target = reward+self.gamma*self.q_table.loc[new_state,:].max()        
        else:
            q_target = reward
        self.q_table.loc[state,action] += self.alpha *(q_target-self.q_table.loc[state,action])


if __name__ =='__main__':
    env = Maze()
    
    
    ACTIONS = ['L', 'R','U','D']     # available actions
    EPSILON = 0.9   # greedy police
    ALPHA = 0.1     # learning rate
    GAMMA = 0.9    # discount factor
    MAX_EPISODES = 13   # maximum episodes
    TERMINAL = 'terminal'
    rlmaze = RLMaze(ACTIONS, ALPHA, GAMMA, EPSILON,  TERMINAL)
    for episode in range(MAX_EPISODES):
        state = env.reset()
        while True:
            env.render()
            action = rlmaze.choose_action(state)
            new_state, reword, done = env.step(action)
            rlmaze.reinforcement_learning(state,action,reward,new_state)
            state = new_state
            if done:
                break
