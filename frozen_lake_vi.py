import mdp_copy
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt
import seaborn as sns

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


    ### 0.9 discount low ep

    vi = mdp_copy.ValueIteration(T, R, 0.9, epsilon=.0000001)
    vi.verbose = True
    vi.max_iter = 1000
    vi.run()
    print("Iterations", vi.iter)
    shape = (30, 30)
    policy_chars = np.array(list(map(lambda a: ACTION_MAP[a], np.array(vi.policy))))
    policy_for_plot = policy_chars.reshape(shape)
    value_for_plot = np.array(vi.V).reshape(shape)
    sns.heatmap(value_for_plot, annot=policy_for_plot, fmt='s')
    plt.savefig('value_policy_9_low_ep.png')
    plt.close()

    time_array = []
    iter_array = []
    value_array = []
    reward_array = []
    count = 1
    for i in vi.run_stats:
        iter_array.append(count)
        time_array.append(i['Time'])
        value_array.append(i['Max V'])
        reward_array.append(i['Error'])
        count = count + 1

    plt.plot(iter_array, time_array, label='Time')
    plt.legend(loc=4, fontsize=8)
    plt.title("Timing vs Iterations Value Iteration")
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_time_9_low_ep.png')
    plt.close()

    plt.plot(iter_array, value_array, label='Max Value')
    plt.legend(loc=4, fontsize=8)
    plt.title("Max Value vs Iterations Value Iteration")
    plt.ylabel('Value')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_max_value_9_low_ep.png')
    plt.close()

    plt.plot(iter_array, reward_array, label='Error')
    plt.legend(loc=4, fontsize=8)
    plt.title("Error vs Iterations Value Iteration")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_Error_9_low_ep.png')
    plt.close()

    ### 0.1 discount low ep

    vi = mdp_copy.ValueIteration(T, R, 0.1, epsilon=.0000001)
    vi.verbose = True
    vi.max_iter = 1000
    vi.run()
    print("Iterations", vi.iter)
    shape = (30, 30)
    policy_chars = np.array(list(map(lambda a: ACTION_MAP[a], np.array(vi.policy))))
    policy_for_plot = policy_chars.reshape(shape)
    value_for_plot = np.array(vi.V).reshape(shape)
    sns.heatmap(value_for_plot, annot=policy_for_plot, fmt='s')
    plt.savefig('value_policy_1_low_ep.png')
    plt.close()

    time_array = []
    iter_array = []
    value_array = []
    reward_array = []
    count = 1
    for i in vi.run_stats:
        iter_array.append(count)
        time_array.append(i['Time'])
        value_array.append(i['Max V'])
        reward_array.append(i['Error'])
        count = count + 1

    plt.plot(iter_array, time_array, label='Time')
    plt.legend(loc=4, fontsize=8)
    plt.title("Timing vs Iterations Value Iteration")
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_time_1_low_ep.png')
    plt.close()

    plt.plot(iter_array, value_array, label='Max Value')
    plt.legend(loc=4, fontsize=8)
    plt.title("Max Value vs Iterations Value Iteration")
    plt.ylabel('Value')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_max_value_1_low_ep.png')
    plt.close()

    plt.plot(iter_array, reward_array, label='Error')
    plt.legend(loc=4, fontsize=8)
    plt.title("Error vs Iterations Value Iteration")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_Error_1_low_ep.png')
    plt.close()

    ### 0.9 discount high ep

    vi = mdp_copy.ValueIteration(T, R, 0.9, epsilon=.01)
    vi.verbose = True
    vi.max_iter = 1000
    vi.run()
    print("Iterations", vi.iter)
    shape = (30, 30)
    policy_chars = np.array(list(map(lambda a: ACTION_MAP[a], np.array(vi.policy))))
    policy_for_plot = policy_chars.reshape(shape)
    value_for_plot = np.array(vi.V).reshape(shape)
    sns.heatmap(value_for_plot, annot=policy_for_plot, fmt='s')
    plt.savefig('value_policy_9_high_ep.png')
    plt.close()

    time_array = []
    iter_array = []
    value_array = []
    reward_array = []
    count = 1
    for i in vi.run_stats:
        iter_array.append(count)
        time_array.append(i['Time'])
        value_array.append(i['Max V'])
        reward_array.append(i['Error'])
        count = count + 1

    plt.plot(iter_array, time_array, label='Time')
    plt.legend(loc=4, fontsize=8)
    plt.title("Timing vs Iterations Value Iteration")
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_time_9_high_ep.png')
    plt.close()

    plt.plot(iter_array, value_array, label='Max Value')
    plt.legend(loc=4, fontsize=8)
    plt.title("Max Value vs Iterations Value Iteration")
    plt.ylabel('Value')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_max_value_9_high_ep.png')
    plt.close()

    plt.plot(iter_array, reward_array, label='Error')
    plt.legend(loc=4, fontsize=8)
    plt.title("Error vs Iterations Value Iteration")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_Error_9_high_ep.png')
    plt.close()

    ### 0.1 discount high ep

    vi = mdp_copy.ValueIteration(T, R, 0.1, epsilon=.01)
    vi.verbose = True
    vi.max_iter = 1000
    vi.run()
    print("Iterations", vi.iter)
    shape = (30, 30)
    policy_chars = np.array(list(map(lambda a: ACTION_MAP[a], np.array(vi.policy))))
    policy_for_plot = policy_chars.reshape(shape)
    value_for_plot = np.array(vi.V).reshape(shape)
    sns.heatmap(value_for_plot, annot=policy_for_plot, fmt='s')
    plt.savefig('value_policy_1_high_ep.png')
    plt.close()

    time_array = []
    iter_array = []
    value_array = []
    reward_array = []
    count = 1
    for i in vi.run_stats:
        iter_array.append(count)
        time_array.append(i['Time'])
        value_array.append(i['Max V'])
        reward_array.append(i['Error'])
        count = count + 1

    plt.plot(iter_array, time_array, label='Time')
    plt.legend(loc=4, fontsize=8)
    plt.title("Timing vs Iterations Value Iteration")
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_time_1_high_ep.png')
    plt.close()

    plt.plot(iter_array, value_array, label='Max Value')
    plt.legend(loc=4, fontsize=8)
    plt.title("Max Value vs Iterations Value Iteration")
    plt.ylabel('Value')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_max_value_1_high_ep.png')
    plt.close()

    plt.plot(iter_array, reward_array, label='Error')
    plt.legend(loc=4, fontsize=8)
    plt.title("Error vs Iterations Value Iteration")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.savefig('value_policy_Error_1_high_ep.png')
    plt.close()
