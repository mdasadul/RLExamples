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
        self.q_table = pd.DataFrame(columns=actions,dtype=np.float64)
        
    def choose_action(self, state):
        self.check_state_exists(state)
        if np.random.uniform() > self.epsilon :
            action_name = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[state,:]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action_name = state_action.idxmax()

        return action_name


    def check_state_exists(self,state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = [0]*len(self.actions)
            

    def reinforcement_learning(self,state,action,reward,new_state,new_action):
        self.check_state_exists(new_state)
        if new_state !=self.terminal:
            q_target = reward+self.gamma*self.q_table.loc[new_state,new_action]        
        else:
            q_target = reward
        self.q_table.loc[state,action] += self.alpha *(q_target-self.q_table.loc[state,action])


if __name__ =='__main__':
    env = Maze()
    
    
    ACTIONS = list(range(env.n_actions))     # available actions
    EPSILON = 0.9   # greedy police
    ALPHA = 0.1     # learning rate
    GAMMA = 0.9    # discount factor
    MAX_EPISODES = 20   # maximum episodes
    TERMINAL = 'terminal'
    rlmaze = RLMaze(ACTIONS, ALPHA, GAMMA, EPSILON,  TERMINAL)
    for episode in range(MAX_EPISODES):
        state = env.reset()
        step=0
        action = rlmaze.choose_action(str(state))
        while True:
            env.render()
            
            new_state, reward, done = env.step(action)
            new_action = rlmaze.choose_action(str(new_state))
            rlmaze.reinforcement_learning(str(state),action,reward,str(new_state),new_action)
            state = new_state
            action = new_action
            step +=1
            if done:
                if reward ==1:
                    print("SUCCESS  after Number of steps %d  in epicode %d", (step,episode))
                else:
                    print("FAIL  after Number of steps %d  in epicode %d", (step,episode))
         
                break
        

