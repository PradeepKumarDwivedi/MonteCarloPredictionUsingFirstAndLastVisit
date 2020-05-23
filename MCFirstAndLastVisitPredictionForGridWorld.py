import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

from gridWorldEnvironment import GridWorld

# creating gridworld environment
gw = GridWorld(gamma = .9, theta = .5)

FirstVisitValues = []
EveryVisitValues = []
FirstVisitIterations = []
EveryVisitIterations = []

def generate_random_episode(env):
    episode = []
    done = False
    current_state = np.random.choice(env.states)
    episode.append((current_state, -1))
    while not done:
        action = np.random.choice(env.actions)
        if current_state == 1 and action == 'L':
            action = np.random.choice(['U', 'D', 'L', 'R'], 1, p=[0.1, 0.1, 0.7, 0.1])
        if current_state == 5:
            if action == 'L' or action == 'U':
                action = np.random.choice(['U', 'D', 'L', 'R'], 1, p=[0.4, 0.1, 0.4, 0.1])
        next_state, reward = gw.state_transition(current_state, action)
        episode.append((next_state, reward))
        if next_state == 0:
            done = True
        current_state = next_state
        print('Running')
    return episode


generate_random_episode(gw)


def value_array(env):
    return np.zeros(len(env.states) + 2)


def first_visit_mc(env, num_iter):
    values = value_array(env)
    returns = dict()
    for state in env.states:
        returns[state] = list()

    for i in range(num_iter):
        episode = generate_random_episode(env)
        already_visited = set({0})  # also exclude terminal state (0)
        for s, r in episode:
            if s not in already_visited:
                already_visited.add(s)
                idx = episode.index((s, r))
                G = 0
                j = 1
                while j + idx < len(episode):
                    G = env.gamma * (G + episode[j + idx][1])
                    FirstVisitValues.append(G)
                    FirstVisitIterations.append(j)
                    j += 1
                returns[s].append(G)
                values[s] = np.mean(returns[s])
    return values, returns

#time
values, returns = first_visit_mc(gw, 70)
# obtained values
values

whichVisit = -1
def show_values(values,whichVisit):
    values = values.reshape(4, 4)
    ax = seaborn.heatmap(values, cmap="Blues_r", annot=True, linecolor="#282828", linewidths=0.1)
    if whichVisit == 1:
       plt.title('First Visit State Value Table')
       plt.savefig('FirstVisitStateValueTable.png')
    elif whichVisit == 2:
        plt.title('Every Visit State Value Table')
        plt.savefig('EveryVisitStateValueTable.png')
    plt.show()


show_values(values,1)
plt.plot(FirstVisitValues, FirstVisitIterations)
plt.xlabel('Values in States')
plt.ylabel('All States')
plt.title('Values VS States(First Visit)')
plt.savefig('Values VS States(First Visit).png')
plt.show()


def every_visit_mc(env, num_iter):
    values = value_array(env)
    returns = dict()
    for state in env.states:
        returns[state] = list()

    for i in range(num_iter):
        episode = generate_random_episode(env)
        for s, r in episode:
            if s != 0:  # exclude terminal state (0)
                idx = episode.index((s, r))
                G = 0
                j = 1
                while j + idx < len(episode):
                    G = env.gamma * (G + episode[j + idx][1])
                    EveryVisitValues.append(G)
                    EveryVisitIterations.append(j)
                    j += 1
                returns[s].append(G)
                values[s] = np.mean(returns[s])
    return values, returns
#time
values, returns = every_visit_mc(gw, 70)

# obtained values
values

show_values(values,2)
plt.plot(EveryVisitValues, EveryVisitIterations)
plt.xlabel('Values in States')
plt.ylabel('All States')
plt.title('Values VS States(Every Visit)')
plt.savefig('Values VS States(Every Visit).png')
plt.show()