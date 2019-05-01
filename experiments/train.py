import json
import os
import numpy as np
import tensorflow as tf
import time
import pickle
# import make_env

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer, DDPGAgentTrainer
import tensorflow.contrib.layers as layers
from maddpg.common.config_args import parse_args, build_summaries

arglist = parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        if scope == "qMA_func" or scope == "target_qMA_func":
            out = layers.fully_connected(out, num_outputs=int(num_units/2), activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=int(num_units/4), activation_fn=tf.nn.relu)
        else:
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

class ActorNetwork():

    def __init__(self, num_agents, batch_size, p_index):
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.p_index = p_index

    def encoder(self, input, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            out = input[self.p_index]
            out = layers.fully_connected(out, num_outputs=arglist.enc_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=arglist.readProj, activation_fn=tf.nn.relu)
        return out

    def read(self, input, memory, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            out = input
            out = layers.fully_connected(out, num_outputs=arglist.readProj, activation_fn=None)
            out = tf.concat([input, out, memory], axis=-1)
            out = layers.fully_connected(out, num_outputs=arglist.writeProj, activation_fn=tf.nn.sigmoid)
            out = memory * out
        return out

    def write(self, input, memory, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.concat([input, memory], axis=-1)
            out = layers.fully_connected(out, num_outputs=arglist.writeProj, activation_fn=tf.nn.tanh)
            rem_out = layers.fully_connected(input, num_outputs=arglist.memUnits, activation_fn=tf.nn.sigmoid)
            forget_out = layers.fully_connected(input, num_outputs=arglist.memUnits, activation_fn=tf.nn.sigmoid)
            mem_out = rem_out * out + forget_out * memory
        return mem_out

    def action(self, input, r_info, memory, num_outputs, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.concat([input, r_info, memory], axis=-1)
            out = layers.fully_connected(out, num_outputs=arglist.action_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

    def adv_model(self, input, memory, num_outputs, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # Get encoding of current observation
            obs_encode = self.encoder(input, "encoder", reuse=reuse)
            read_info = self.read(obs_encode, memory, "read_unit", reuse=reuse)
            memory_new = self.write(obs_encode, memory, "write_unit", reuse=reuse)
            output = self.action(obs_encode, read_info, memory_new,  num_outputs, "action_cell", reuse=reuse)
        return output, memory_new


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
        print("yes")
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    adv_model = ActorNetwork
    advTrainer = MADDPGAgentTrainer
    goodTrainer = DDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(advTrainer(
            "adv_agent", adv_model, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(goodTrainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def train(arglist):
    # Create experiment folder
    tensorboard_dir = os.path.join('./exp_data', arglist.exp_name, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    # build summaries
    summary_ops, summary_vars = build_summaries(4)
    with U.single_threaded_session() as sess:
        # Create environment
        np.random.seed(arglist.random_seed)
        tf.set_random_seed(arglist.random_seed)
        env = make_env(arglist.scenario,  arglist.benchmark)
        env.seed(arglist.random_seed)
        # Create agent trainers
        print(env.n)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()
        if not arglist.benchmark or arglist.restore:
            writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            writer.flush()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = os.path.join('./exp_data/' + arglist.exp_name + arglist.save_dir + '60000')
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        # Intialize memory
        memory_init = np.random.normal(loc=0.0, scale=1.0, size=(1, arglist.memUnits))

        print('Starting iterations...')
        while True:
            # Reset memory on new episode
            if episode_step == 0:
                memory_state_in = memory_init

            # Populate actions, states for all agents
            action_n = []
            memory_a = []

            for i, agent in enumerate(trainers):
                if i < num_adversaries:
                    action_n.append(agent.action(obs_n, memory_state_in))
                    memory_state_in = agent.memory_state(obs_n, memory_state_in)
                    memory_state_in = np.expand_dims(memory_state_in, axis=0)
                    memory_a.append(memory_state_in)
                else:
                    action_n.append(agent.act(obs_n[i]))
            # print(np.shape(control_a[0]), "control A agent 0")
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                if i < num_adversaries:
                    agent.experience(obs_n[i], memory_a[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                else:
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    benchmark_dir = os.path.join('./exp_data', arglist.exp_name, arglist.benchmark_dir)
                    if not os.path.exists(benchmark_dir):
                        os.mkdir(benchmark_dir)
                    file_name = './exp_data/' + arglist.exp_name + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                # Policy File
                save_dir = './exp_data/' + arglist.exp_name + arglist.save_dir + str(len(episode_rewards))
                U.save_state(save_dir, saver=saver)
                # Tensorboard
                ep_reward = [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards]
                if not arglist.benchmark or arglist.restore:
                    summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: ep_reward[0],
                                                                   summary_vars[1]: ep_reward[1]
                                                                   # summary_vars[2]: ep_reward[2],
                                                                   })

                    writer.add_summary(summary_str, len(episode_rewards))
                    writer.flush()
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                plot_dir = os.path.join('./exp_data', arglist.exp_name, arglist.plots_dir)
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
                rew_file_name = './exp_data/' + arglist.exp_name + '/' + arglist.plots_dir + '/' + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = './exp_data/' + arglist.exp_name + '/' + arglist.plots_dir + '/' + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

def benchmark(arglist):

    bench_file = './exp_data/' + arglist.exp_name + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.pkl'
    bench_load = open(bench_file, 'rb')
    bench_val = pickle.load(bench_load)
    print(bench_val)
    bench_load.close()

    collision = 0
    min_dist = 0
    occ_landmarks = 0
    num_steps = 0
    reward = 0
    for ep in range(len(bench_val)):
        done = False
        steps = 0
        while not done:
            if steps < len(bench_val[ep][0]):
                for agent in range(len(bench_val[ep][0][steps])):
                    if not steps == 0:
                        # print(steps)
                        if bench_val[ep][0][steps][agent][3] == arglist.num_adversaries: # Assuming num landmarks = num adversaries
                            reward += bench_val[ep][0][steps][agent][0]
                            collision += bench_val[ep][0][steps][agent][1] - bench_val[ep][0][0][agent][1]   # Account for the intial collision at step 0, due to randomly generated environment
                            min_dist += bench_val[ep][0][steps][agent][2]
                            occ_landmarks += bench_val[ep][0][steps][agent][3]
                            num_steps += steps
                            done = True
                        elif steps == (len(bench_val[ep][0]) - 1):
                            reward += bench_val[ep][0][steps][agent][0]
                            collision += bench_val[ep][0][steps][agent][1] - bench_val[ep][0][0][agent][1]  # Account for the intial collision at step 0, due to randomly generated environment
                            min_dist += bench_val[ep][0][steps][agent][2]
                            occ_landmarks += bench_val[ep][0][steps][agent][3]
                            num_steps += steps
                            done = True
                steps += 1
            else:
                break

    reward = reward / (len(bench_val) * len(bench_val[0][0][0]))
    collision = collision / (len(bench_val) * len(bench_val[0][0][0]))
    min_dist = min_dist / (len(bench_val) * len(bench_val[0][0][0]))
    occ_landmarks = occ_landmarks / (len(bench_val) * len(bench_val[0][0][0]))
    num_steps = num_steps / (len(bench_val) * len(bench_val[0][0][0]))

    print("Average Reward: ", reward)
    print("Average Collisions: ", collision)
    print("Average Dist: ", min_dist)
    print("Average Occupied Landmarks: ", occ_landmarks)
    print("Average Steps: ", num_steps)

    save_info_dir = './exp_data/' + arglist.exp_name + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.txt'
    with open(save_info_dir, 'w') as bench_file:
        bench_file.write("Model Name: %s" % arglist)
        bench_file.write("Average Reward: %d" % reward)
        bench_file.write("Average Collisions: %d" % collision)
        bench_file.write("Average Dist: %d" % min_dist)
        bench_file.write("Average Occupied Landmarks: %d" % occ_landmarks)
        bench_file.write("Average Steps: %d" % num_steps)

if __name__ == '__main__':
    exp_dir = os.path.join('./exp_data', arglist.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not arglist.benchmark or arglist.restore or arglist.display:
        with open('./exp_data/' + arglist.exp_name + '/args.txt', 'w') as fp:
            json.dump(arglist.__dict__, fp, indent=2)

    train(arglist)

    if arglist.benchmark:
        benchmark(arglist)