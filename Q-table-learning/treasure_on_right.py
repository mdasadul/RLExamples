import numpy as np
import pandas as pd
import time

np.random.seed(20)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['L', 'R']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move
TERMINAL = 'T'


def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states,len(actions))),
    columns = actions,)
    return table


def choose_action(state, q_table):
    state_actions  = q_table.iloc[state,:]
    if np.random.uniform() > EPSILON or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    
    return action_name


def get_env_feedback(S, A):
    if A =='R':
        if S == N_STATES-2:
            new_S = TERMINAL
            R = 1
        else:
            new_S = S + 1
            R = 0
    else:
        if S==0:
            new_S = S
            R = 0
        else:
            new_S = S - 1
            R = 0
    return new_S, R



def display_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == TERMINAL:
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'A'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def reinforcement_learning():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step = 0
        S = 0
        display_env(S,episode, step)
        done = False
        while not done:
            A = choose_action(S,q_table)
            new_S, R = get_env_feedback(S, A)
            if new_S != TERMINAL:
                q_target = R+GAMMA*q_table.loc[new_S,:].max()
                
            else:
                
                q_target = R
                done = True
            q_table.loc[S,A] += ALPHA *(q_target-q_table.loc[S,A])
            S = new_S
            step +=1
            display_env(S,episode, step)
    return q_table


if __name__ == "__main__":
    q_table = reinforcement_learning()
    print('\r\nQ-table:\n')
    print(q_table)