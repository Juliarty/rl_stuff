"""This is the example module.

This module does stuff.
"""

__all__ = ['SimpleGridWorld']
__version__ = '0.1'
__author__ = 'Juliarty'

import numpy as np
import random
import plotly.express as px


# 1-D array where you can move to left or right.
# The thing of this world that the only where you can get reward is a terminal state, otherwise -1.
class SimpleGridWorld:
    actions_num = 2
    _actions = ['left', 'right']

    def __init__(self, terminal_state_pos=3, states_num=5):
        if terminal_state_pos >= states_num:
            raise AttributeError("'terminal_state_pos' is to be less than 'states_num'")

        self.states_num = states_num + 1  # added end of episode (it's the last state)
        self._states = range(self.states_num)
        self.terminal_state_pos = terminal_state_pos
        self.end_of_episode_state = states_num

    def get_model(self):
        # move to the left
        dynamics_left = np.zeros(shape=(self.states_num, self.states_num))
        dynamics_right = np.zeros(shape=(self.states_num, self.states_num))
        for i in range(self.states_num - 1):    # Don't include last state (end of game)
            # terminal state
            if i == self.terminal_state_pos:
                dynamics_left[i, self.end_of_episode_state] = 1
                dynamics_right[i, self.end_of_episode_state] = 1
                continue

            dynamics_left[i, self._states[i - 1 if i > 0 else i]] = 1
            dynamics_right[i, self._states[i + 1 if i != self.states_num - 2 else i]] = 1

        return np.asarray([dynamics_left, dynamics_right])

    def get_reward(self):
        reward_mx = np.full(shape=(self.states_num, self.actions_num), fill_value=-1, dtype=int)
        reward_mx[self.end_of_episode_state, :] = np.zeros(self.actions_num)
        reward_mx[self.terminal_state_pos, :] = np.ones(self.actions_num)
        return reward_mx

    def get_simple_policy(self):
        policy_mx = np.zeros(shape=(self.states_num, self.actions_num))
        for i in range(self.states_num):
            policy_mx[i, i % 2] = 1
        return policy_mx

    def print_nice_q(self, q_mx):
        out_format = "Position: {pos}; Left: {left_reward}; Right: {right_reward};"
        for i in range(self.states_num):
            print(out_format.format(pos=i, left_reward=q_mx[i, 0], right_reward=q_mx[i, 1]))

    def print_nice_policy(self, policy):
        out_format = "Position: {pos}; Action: {Action};"
        for i in range(self.states_num):
            action = "Go Left" if policy[i, 0] == 1 else "Go Right"
            print(out_format.format(pos=i, action=action))

    def draw_plot(self, policy):
        fig = px.scatter(x=self._states[:-1], y=np.argmax(policy[:-1], axis=1))
        fig.show()
