"""This is the example module.

This module does stuff.
"""

__all__ = ['BjStuff']
__version__ = '0.1'
__author__ = 'Juliarty'

import numpy as np
import random


def get_dealer_sum_possibilities(num_of_exp=1000000, card_num_dealer_takes=6):
    card_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4

    card_to_sum2prob = {}
    for card in card_values:
        card_to_sum2prob[card] = {}
        random.seed()
        for i in range(num_of_exp):
            if card == 1:
                usable_ace = True
                dealer_sum = 11
            else:
                usable_ace = False
                dealer_sum = card

            sample = random.sample(all_cards, card_num_dealer_takes)
            j = 0
            while dealer_sum < 17 and j < card_num_dealer_takes:
                current_card = sample[j]

                if current_card == 1:
                    if dealer_sum + 11 < 22:
                        usable_ace = True
                        dealer_sum += 11
                    else:
                        dealer_sum += 1

                dealer_sum += current_card

                if dealer_sum > 21 and usable_ace:
                    usable_ace = False
                    dealer_sum -= 10

                if dealer_sum > 21:
                    dealer_sum = 22
                    break
                j += 1

            if dealer_sum not in card_to_sum2prob[card].keys():
                card_to_sum2prob[card][dealer_sum] = 0

            card_to_sum2prob[card][dealer_sum] += 1

    for card in card_to_sum2prob.keys():
        for s in card_to_sum2prob[card].keys():
            card_to_sum2prob[card][s] /= num_of_exp

    return card_to_sum2prob


class BjStuff:
    # state is a vector (usable_ace, player_sum, dealer_first_card)
    # also there is a player_sum 'busted' for the player
    _dealer_card_num = 10  # [1, 2 , 3 , ..., 10]
    _player_sum_num = 11  # [12, ... , 21, 22 = 'busted']
    _usable_ace_num = 2  # [True, False]

    states_num = _dealer_card_num * _player_sum_num * _usable_ace_num
    # stick = 0; hit = 1
    actions_num = 2

    _state_to_num = {}
    _num_to_state = {}

    # Probabilities of dealer's card sum (num of experiments = 1,000,000)
    _first_card_to_dealer_sum = \
        {1: {22: 0.11619, 20: 0.13008, 21: 0.35846, 17: 0.126481, 19: 0.127426, 18: 0.141363},
         2: {22: 0.35268, 18: 0.141493, 21: 0.118114, 17: 0.137443, 20: 0.120139, 19: 0.130124, 15: 1e-06, 16: 6e-06},
         3: {22: 0.370759, 20: 0.119338, 18: 0.138101, 19: 0.120959, 21: 0.112276, 17: 0.138566, 16: 1e-06},
         4: {18: 0.140164, 19: 0.122578, 22: 0.39749, 20: 0.114488, 21: 0.11257, 17: 0.11271},
         5: {19: 0.104618, 17: 0.209603, 21: 0.095617, 22: 0.376998, 18: 0.110758, 20: 0.102406},
         6: {18: 0.209806, 22: 0.42078, 20: 0.100465, 21: 0.098317, 19: 0.10081, 17: 0.069822},
         7: {19: 0.154573, 17: 0.365162, 21: 0.072778, 22: 0.262557, 20: 0.073691, 18: 0.071239},
         8: {21: 0.063999, 18: 0.366757, 20: 0.144998, 22: 0.244215, 17: 0.128096, 19: 0.051935},
         9: {17: 0.117423, 19: 0.348992, 18: 0.128412, 21: 0.135482, 22: 0.227042, 20: 0.042649},
         10: {20: 0.348054, 18: 0.125723, 19: 0.118627, 17: 0.118406, 22: 0.246888, 21: 0.042302}}

    def __init__(self):
        ii = 0
        for i in range(self._usable_ace_num):
            for j in range(self._player_sum_num):
                for k in range(self._dealer_card_num):
                    self._state_to_num[(i, j + 12, k + 1)] = ii
                    self._num_to_state[ii] = (i, j + 12, k + 1)
                    ii += 1

    # R is a matrix of size m x n,
    # which consists of float numbers between -1.0 and 1.0
    # m - number of states, n - number of actions
    def get_reward(self):
        m = self.states_num
        n = self.actions_num

        # no matter what, if we are 'busted' we get a negative reward
        # hit; we don't get any reward

        # stick = 0; we get a reward only if dealers sum is less than ours
        reward_matrix = np.zeros([m, n])
        for i in range(self._usable_ace_num):
            for j in range(self._player_sum_num):
                for k in range(self._dealer_card_num):
                    state = self._state_to_num[(i, j + 12, k + 1)]
                    if j + 12 == 22:
                        reward_matrix[state][0] = -1
                        reward_matrix[state][1] = -1
                        continue

                    for dealer_sum in self._first_card_to_dealer_sum[k + 1].keys():
                        if dealer_sum == 22:
                            reward_matrix[state][0] += self._first_card_to_dealer_sum[k + 1][dealer_sum]
                            continue

                        if j + 12 < dealer_sum:
                            reward_matrix[state][0] -= \
                                self._first_card_to_dealer_sum[k + 1][dealer_sum]
                        elif j + 12 > dealer_sum:
                            reward_matrix[state][0] += \
                                self._first_card_to_dealer_sum[k + 1][dealer_sum]

        return reward_matrix

    # Model matrix shows the dynamic of a game
    # each row corresponds to a state where we begin from,
    # each column corresponds to a state where the environment will be
    # after we take the action.
    def get_model(self):
        cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        dynamics_matrix = np.zeros(shape=(self.states_num, self.states_num))
        for i in range(self._usable_ace_num):
            for j in range(self._player_sum_num):
                for k in range(self._dealer_card_num):
                    for card in cards:
                        dealer_card = k + 1
                        usable_ace = i
                        player_sum = j + 12
                        s1 = self._state_to_num[(usable_ace, player_sum, dealer_card)]

                        next_usable_ace = usable_ace
                        next_player_sum = 0

                        # there is a usable ace
                        if usable_ace == 0:
                            if card == 1 and 11 + player_sum < 22:
                                next_usable_ace = 1
                                next_player_sum = player_sum + 11
                                continue
                            next_usable_ace = 0
                            next_player_sum = player_sum + card
                            if next_player_sum > 21:    # 'busted'
                                next_player_sum = 22
                        else:
                            next_player_sum = player_sum + card
                            if next_player_sum > 21:
                                next_player_sum -= 10
                                next_usable_ace = 0

                        s2 = self._state_to_num[(next_usable_ace, next_player_sum, dealer_card)]
                        dynamics_matrix[s1][s2] += 4/52
        return np.asarray([np.identity(self.states_num), dynamics_matrix])

    def get_state_num(self, usable_ace, player_sum, dealer_card):
        return self._state_to_num[(usable_ace, player_sum, dealer_card)]

    # Returns a state = (usable_ace, player_sum, dealer_first_card).
    def get_state(self, num_of_state):
        return self._num_to_state[num_of_state]

    def get_simple_policy(self, stick_after=17):
        policy_mx = np.zeros(shape=(self.states_num, self.actions_num))
        for i in range(self._usable_ace_num):
            for j in range(self._player_sum_num):
                for k in range(self._dealer_card_num):
                    if j + 12 >= 17:
                        # always stick
                        policy_mx[self._state_to_num[(i, j + 12, k + 1)], 0] = 1
                    else:
                        # hit
                        policy_mx[self._state_to_num[(i, j + 12, k + 1)], 1] = 1
        return policy_mx

    def print_nice_q(self, q_mx):
        for i in range(self._usable_ace_num):
            for j in range(self._player_sum_num):
                for k in range(self._dealer_card_num):
                    print("Usable ace: ", i, "; " 
                          "Player's sum: ", j + 12, "; ",
                          "Dealer's card: ", k + 1, ";",
                          "Stick: ", q_mx[self._state_to_num[(i, j + 12, k + 1)], 0], ",",
                          "Hit: ", q_mx[self._state_to_num[(i, j + 12, k + 1)], 1])


if __name__ == "__main__":
    s = BjStuff()
    model = s.get_model()
    print(s.get_reward())
    # print(get_dealer_sum_possibilities())
