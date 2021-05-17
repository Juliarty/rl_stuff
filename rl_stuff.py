import numpy as np
import random
import blackjack_stuff as bjs


# Dynamic programming

# state x actions = m x n
# R(s_t) - immediate reward, m x 1
# dynamic_mx - dynamics matrix (shape = (actions_num, states_num, states_num))
# PI - policy (probability of taking the action a in the state s), m x n
# df - discount factor
# Q - action-value function (table), m x n
def policy_evaluation_q(states_num,
                        actions_num,
                        dynamics_mx,
                        policy_mx,
                        reward_mx,
                        discount_factor,
                        num_of_iteration=2):
    # according to Bellman Equation
    q_mx = np.zeros(shape=[states_num, actions_num])

    # Expected reward depends on policy and action-value function.
    # Here, I'm trying to get a vector of expected rewards (1-D vector of shape=(states_num, 1)).
    # discount_factor * (PI \element-wise-dot Q) * 1
    ones = discount_factor * np.ones(shape=[actions_num, 1])
    for i in range(num_of_iteration):
        expected_reward = np.multiply(policy_mx, q_mx) @ ones
        for j in range(actions_num):
            q_mx[:, j] = reward_mx[:, j] + (dynamics_mx[j, :, :] @ expected_reward).flatten()

    return q_mx


# Monte-Carlo


# TD(0)

def update_eligibility_trace(tr_array, discount_factor, lamb, state):
    tmp = discount_factor * lamb
    tr_array *= tmp
    tr_array[state] += 1


if __name__ == '__main__':
    bj = bjs.BjStuff()
    q_mx = policy_evaluation_q(bj.states_num, bj.actions_num,
                               bj.get_model(), bj.get_simple_policy(),
                               bj.get_reward(), 1, num_of_iteration=100)

    bj.print_nice_q(q_mx)
