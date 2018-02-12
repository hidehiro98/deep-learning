import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=0.005, alpha=0.3, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # return np.random.choice(self.nA)
        self.epsilon = 1.0 / ((i_episode / 800) + 1)

        policy = np.ones(self.nA) * self.epsilon / self.nA
        policy[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon / self.nA
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action, reward, next_state, done, i_episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.Q[state][action] += 1
        self.epsilon = 1.0 / ((i_episode / 800) + 1)

        next_policy = np.ones(self.nA) * self.epsilon / self.nA
        next_policy[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon / self.nA

        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.sum(self.Q[next_state] * next_policy) - self.Q[state][action])
