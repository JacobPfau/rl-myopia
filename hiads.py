import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

ACTION_SPACE = [0, 1] # [DONT PUSH, PUSH]

def act(state, action_values, epsilon):
    """
    args:
        action_values: array of q-values for actions
        
    returns:
        action, one of 1 (push) or -1 (don't push)
    """
    greedy_action = ACTION_SPACE[np.argmax(action_values[state])]
    random_action = np.random.choice(ACTION_SPACE)
    return np.random.choice([random_action, greedy_action], p=[epsilon, 1-epsilon])


def reward(state, action):
    return -action + 10 * state


def run_episode(state, action_values, epsilon=0.1):
    a = act(state,action_values,epsilon)
    r = reward(state, a)
    return a, r


def update_running_mean(num_values, current_mean, next_value):
    return (num_values * current_mean + next_values) / (num_values + 1)
    

def append_to_log(dct, update_dict):
    """appends update_dict to dict, entry-wise. Creates list entry
    if it doesn't exist.
    """
    for key, newvalue in update_dict.items():
        dct.setdefault(key, []).append(newvalue)
    return dct
    

class QLearning:
    def __init__(self, q_init = {0:[0,0],1:[0,0]}, schedule: callable = lambda n: 1/n):
        """
        args:
            schedule: 1/n corresponds to computing the average of 
        empirical q-values. Constant step size in (0, 1) corresponds 
        to taking a weighted average instead, with values close to 1
        weighting recent values more.
        """
        self.action_values = q_init
        self.num_updates = [1, 1]
        self.rundata = {}
        self.weighted_avg = True
        self.schedule = schedule
    
    def log(self, **data):
        append_to_log(self.rundata, data)
    
    def update(self, state, action, reward):
        """keep track of mean action-values"""
        stepsize = self.schedule(self.num_updates[action])
        self.action_values[state][action] += stepsize * (reward - self.action_values[state][action])
        if state==1 and action==1: print(reward - self.action_values[state][action])
        self.num_updates[action] += 1
        self.log(rewards=reward,
                 actions=action,
                 push_values_0=self.action_values[0][1],
                 dp_values_0=self.action_values[0][0],
                 push_values_1=self.action_values[1][1],
                 dp_values_1=self.action_values[1][0],
                 )

        
def running_mean(a, window=30):
    return np.convolve(a, np.ones(window)/window, mode='valid')



def run_and_plot():
    batch_size = 5 #converges for b.s. = 1000
    epochs = 500
    init_q = {0:[1,1],1:[5,1]}
    qvalues = QLearning(q_init=init_q, schedule=lambda n: 0.1) # deceptive apparent convergence for lr = 0.01.
    eps = 0.15*2 # parameter for epsilon-greedy q-learning
    print(f"Training agent using {eps}-greedy Q-learning...")

    for _ in tqdm(range(epochs)):
        if np.random.uniform()>0.5: a=1
        else: a=0

        batch = [(a,0)]
        for _ in range(batch_size):
            a,r = run_episode(a,qvalues.action_values, epsilon=eps)
            batch.append((a,r))
        for ind,b in enumerate(batch[1:], start=1):
            qvalues.update(batch[ind-1][0],b[0],b[1])



    # plotting
    fig, axs = plt.subplots(3, 1, figsize=[12, 12])
    axs = axs.flatten()

    ax = axs[0]
    ax.plot(qvalues.rundata['push_values_0'], label='push')
    ax.plot(qvalues.rundata['dp_values_0'], label='dont push')
    ax.set_ylabel("learned action-value")
    ax.set_xlabel("training iteration")
    ax.set_title("Q-values during training in didn't push state")
    # ax.set_ylim([7,9])
    # ax.set_xlim([0,500])

    ax.legend()

    ax = axs[1]
    ax.plot(qvalues.rundata['push_values_1'], label='push')
    ax.plot(qvalues.rundata['dp_values_1'], label='dont push')
    ax.set_ylabel("learned action-value")
    ax.set_xlabel("training iteration")
    # ax.set_title("Q-values during training")
    # ax.set_ylim([7,9])
    # ax.set_xlim([0,500])

    ax.legend()


    ax = axs[2]
    push_freq = sum(qvalues.rundata['actions']) / len(qvalues.rundata['actions'])
    push_freq

    ax.bar(["don't push", "push"], [1-push_freq, push_freq])
    ax.set_ylabel("Frequency")
    ax.text(0, 1 - push_freq, f"{round(1-push_freq, 3)}", ha='center', va='bottom')
    ax.text(1, push_freq, f"{round(push_freq, 3)}", ha='center', va='bottom')
    # ax.set_title("Actions taken during all of training")

    plt.show()

if __name__=="__main__":
    run_and_plot()
