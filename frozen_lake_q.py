import mdp_copy
import numpy as np, gym, mdptoolbox as mtools
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym import wrappers
from matplotlib import pyplot as plt
import seaborn as sns
import time

ACTION_MAP = ['<', 'V', '>', '^']

if __name__ == '__main__':
    np.random.seed(300)
    grid_size = 30
    random_map = generate_random_map(size=grid_size)
    env = gym.make("FrozenLake-v0", desc=random_map)
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    T = np.zeros((action_space, observation_space, observation_space))
    R = np.zeros((observation_space))
    for state in env.env.P.keys():
        choices = env.env.P[state]
        for action in choices.keys():
            outcomes = choices[action]
            for outcome in outcomes:
                prob, next_state, reward, terminal = outcome
                T[action][state][next_state] += prob
                if not terminal or state != next_state:
                    R[next_state] = reward
    R.reshape((grid_size, grid_size))

    ### 0.9 discount 0.1 alpha

    q_results = mdp_copy.QLearning(T, R, 0.99, n_iter=2000000, epsilon=1.0, alpha=0.1)
    q_results.verbose = True
    q_results.run()
    shape = (30, 30)
    policy_chars = np.array(list(map(lambda a: ACTION_MAP[a], np.array(q_results.policy))))
    policy_for_plot = policy_chars.reshape(shape)
    value_for_plot = np.array(q_results.V).reshape(shape)
    sns.heatmap(value_for_plot, annot=policy_for_plot, fmt='s')
    plt.savefig('frozen_q_learning_9_1_alpha.png')
    plt.close()

    time_array = []
    iter_array = []
    value_array = []
    error_array = []
    reward_array = []
    count = 1
    for i in q_results.run_stats:
        iter_array.append(count)
        time_array.append(i['Time'])
        value_array.append(i['Max V'])
        error_array.append(i['Error'])
        reward_array.append(i['TotalReward'])
        count = count + 1

    plt.plot(iter_array, time_array, label='Time')
    plt.legend(loc=4, fontsize=8)
    plt.title("Timing vs Iterations Value Iteration")
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_time_9_01alpha.png')
    plt.close()

    plt.plot(iter_array, value_array, label='Max Value')
    plt.legend(loc=4, fontsize=8)
    plt.title("Max Value vs Iterations Value Iteration")
    plt.ylabel('Value')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_max_value_9_01alpha.png')
    plt.close()

    plt.plot(iter_array, error_array, label='Error')
    plt.legend(loc=4, fontsize=8)
    plt.title("Error vs Iterations Value Iteration")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_error_901alpha.png')
    plt.close()

    plt.plot(iter_array, reward_array, label='Total Reward')
    plt.legend(loc=4, fontsize=8)
    plt.title("Total Reward vs Iterations Value Iteration")
    plt.ylabel('Total Reward')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_total_reward_901alpha.png')
    plt.close()

    ### 0.9 discount 0.1 alpha

    q_results = mdp_copy.QLearning(T, R, 0.99, n_iter=2000000, epsilon=1.0, alpha=0.99)
    q_results.verbose = True
    q_results.run()

    shape = (30, 30)
    policy_chars = np.array(list(map(lambda a: ACTION_MAP[a], np.array(q_results.policy))))
    policy_for_plot = policy_chars.reshape(shape)
    value_for_plot = np.array(q_results.V).reshape(shape)
    sns.heatmap(value_for_plot, annot=policy_for_plot, fmt='s')
    plt.savefig('frozen_q_learning_99_99_alpha.png')
    plt.close()

    time_array = []
    iter_array = []
    value_array = []
    error_array = []
    reward_array = []
    count = 1
    for i in q_results.run_stats:
        iter_array.append(count)
        time_array.append(i['Time'])
        value_array.append(i['Max V'])
        error_array.append(i['Error'])
        reward_array.append(i['TotalReward'])
        count = count + 1

    plt.plot(iter_array, time_array, label='Time')
    plt.legend(loc=4, fontsize=8)
    plt.title("Timing vs Iterations Value Iteration")
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_time_9_99_alpha.png')
    plt.close()

    plt.plot(iter_array, value_array, label='Max Value')
    plt.legend(loc=4, fontsize=8)
    plt.title("Max Value vs Iterations Value Iteration")
    plt.ylabel('Value')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_max_value_9_99_alpha.png')
    plt.close()

    plt.plot(iter_array, error_array, label='Error')
    plt.legend(loc=4, fontsize=8)
    plt.title("Error vs Iterations Value Iteration")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_error_9_99_alpha.png')
    plt.close()

    plt.plot(iter_array, reward_array, label='Total Reward')
    plt.legend(loc=4, fontsize=8)
    plt.title("Total Reward vs Iterations Value Iteration")
    plt.ylabel('Total Reward')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_total_reward_9_99_alpha.png')
    plt.close()

    ## 0.1 discount 0.1 alpha

    q_results = mdp_copy.QLearning(T, R, 0.1, n_iter=2000000, epsilon=1.0, alpha=0.99)
    q_results.verbose = True
    q_results.run()

    shape = (30, 30)
    policy_chars = np.array(list(map(lambda a: ACTION_MAP[a], np.array(q_results.policy))))
    policy_for_plot = policy_chars.reshape(shape)
    value_for_plot = np.array(q_results.V).reshape(shape)
    sns.heatmap(value_for_plot, annot=policy_for_plot, fmt='s')
    plt.savefig('frozen_q_learning_1_99_alpha.png')
    plt.close()

    time_array = []
    iter_array = []
    value_array = []
    error_array = []
    reward_array = []
    count = 1
    for i in q_results.run_stats:
        iter_array.append(count)
        time_array.append(i['Time'])
        value_array.append(i['Max V'])
        error_array.append(i['Error'])
        reward_array.append(i['TotalReward'])
        count = count + 1

    plt.plot(iter_array, time_array, label='Time')
    plt.legend(loc=4, fontsize=8)
    plt.title("Timing vs Iterations Value Iteration")
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_time_1_99_alpha.png')
    plt.close()

    plt.plot(iter_array, value_array, label='Max Value')
    plt.legend(loc=4, fontsize=8)
    plt.title("Max Value vs Iterations Value Iteration")
    plt.ylabel('Value')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_max_value_1_99_alpha.png')
    plt.close()

    plt.plot(iter_array, error_array, label='Error')
    plt.legend(loc=4, fontsize=8)
    plt.title("Error vs Iterations Value Iteration")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_error_1_99_alpha.png')
    plt.close()

    plt.plot(iter_array, reward_array, label='Total Reward')
    plt.legend(loc=4, fontsize=8)
    plt.title("Total Reward vs Iterations Value Iteration")
    plt.ylabel('Total Reward')
    plt.xlabel('Iterations')
    plt.savefig('frozen_q_learning_total_reward_1_99_alpha.png')
    plt.close()





