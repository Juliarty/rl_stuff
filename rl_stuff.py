import numpy as np
import random
import blackjack_stuff as bjs


# Dynamic programming

# state x actions = m x n
# R(s_t) - immediate reward, m x 1
# dynamic_mx - dynamics matrix (shape = (actions_num, states_num, states_num))
# PI - policy (probability of taking the action a in the state s), m x n
# df - discount factor
# q_mx - action-value function (table), m x n
def policy_evaluation_q(states_num,
                        actions_num,
                        dynamics_mx,
                        policy_mx,
                        reward_mx,
                        discount_factor,
                        num_of_iteration=100,
                        end_condition_func=None):
    # according to Bellman Equation
    q_mx = np.zeros(shape=[states_num, actions_num])
    prev_q_mx = np.zeros(shape=[states_num, actions_num])
    # Expected reward depends on policy and action-value function.
    # Here, I'm trying to get a vector of expected rewards (1-D vector of shape=(states_num, 1)).
    # discount_factor * (PI \element-wise-dot Q) * 1
    ones = discount_factor * np.ones(shape=[actions_num, 1])
    for i in range(num_of_iteration):
        expected_reward = np.multiply(policy_mx, q_mx) @ ones
        for j in range(actions_num):
            q_mx[:, j] = reward_mx[:, j] + (dynamics_mx[j, :, :] @ expected_reward).flatten()

        if end_condition_func is None:
            continue
        if end_condition_func(q_mx, prev_q_mx):
            break
        prev_q_mx = prev_q_mx * 0
        prev_q_mx += q_mx

    return q_mx


def policy_iteration(states_num,
                     actions_num,
                     dynamics_mx,
                     policy_mx,
                     reward_mx,
                     discount_factor=1,
                     greedy_factor=0.8,
                     max_policy_iteration_num=100,
                     max_policy_eval_num=100,
                     policy_eval_stop_criteria=None,
                     policy_iter_stop_criteria=None):

    current_policy = policy_mx
    for i in range(max_policy_iteration_num):
        q_mx = policy_evaluation_q(states_num, actions_num,
                                   dynamics_mx, bj.get_simple_policy(),
                                   reward_mx, discount_factor, num_of_iteration=max_policy_eval_num,
                                   end_condition_func=policy_eval_stop_criteria)
        new_policy = policy_eps_greedy_improvement(current_policy, q_mx, greedy_factor)

        if policy_iter_stop_criteria is not None and \
                policy_iter_stop_criteria(new_policy, current_policy):
            print("Number of iterations: ", i)
            return new_policy

        current_policy = new_policy

    return current_policy


def policy_eps_greedy_improvement(policy_mx, policy_q_mx, eps):
    new_policy = np.zeros(shape=policy_mx.shape)
    for i in range(policy_mx.shape[0]):
        if np.random.binomial(1, 1 - eps):
            action_num = policy_q_mx[i].argmax(0)
        else:
            action_num = np.random.choice(range(policy_mx.shape[1]))
        new_policy[i, action_num] = 1

    return new_policy

# Monte-Carlo


# TD(0)

def update_eligibility_trace(tr_array, discount_factor, lamb, state):
    tmp = discount_factor * lamb
    tr_array *= tmp
    tr_array[state] += 1


def ssq(eps):
    def func(q_mx, prev_q_mx):
        value = np.sum((q_mx - prev_q_mx) ** 2)
        return True if value < eps else False
    return func


if __name__ == '__main__':
    bj = bjs.BjStuff()

    optimal_policy = policy_iteration(bj.states_num,
                                      bj.actions_num,
                                      bj.get_model(),
                                      bj.get_simple_policy(),
                                      bj.get_reward(),
                                      discount_factor=1,
                                      greedy_factor=0.85,
                                      max_policy_iteration_num=1000,
                                      max_policy_eval_num=100,
                                      policy_eval_stop_criteria=ssq(0.00001),
                                      policy_iter_stop_criteria=ssq(0.001))

    q_mx = policy_evaluation_q(bj.states_num, bj.actions_num,
                               bj.get_model(), bj.get_simple_policy(),
                               bj.get_reward(), 1, num_of_iteration=100,
                               end_condition_func=ssq(0.0000001))

    # bj.print_nice_q(q_mx)
    bj.draw_plot(optimal_policy)