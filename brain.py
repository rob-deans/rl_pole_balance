import gym
from collections import deque
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import spline


class Model:
    def __init__(self):
        self.input_states = tf.placeholder(np.float32, shape=[None, 4], name='input')
        self.input_actions = tf.placeholder(np.float32, shape=[None, 2], name='q_values_new')
        self.rewards = tf.placeholder(np.float32, shape=[None], name='rewards')
        self.learning_rate = 1e-3

        net = self.input_states
        init = tf.truncated_normal_initializer()

        net = tf.layers.dense(inputs=net, units=10, activation=tf.nn.relu, kernel_initializer=init, name='dense_1')
        net = tf.layers.dense(inputs=net, units=15, activation=tf.nn.relu, kernel_initializer=init, name='dense_2')
        net = tf.layers.dense(inputs=net, units=10, activation=tf.nn.relu, kernel_initializer=init, name='dense_3')
        net = tf.layers.dense(inputs=net, units=2, kernel_initializer=init, activation=None, name='output')

        self.output = net

        q_reward = tf.reduce_sum(tf.multiply(self.output, self.input_actions), 1)
        loss = tf.reduce_mean(tf.squared_difference(self.rewards, q_reward))
        self.optimiser = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_batch(self, states):
        return self.session.run(self.output, feed_dict={self.input_states: states})

    def predict(self, state):
        return self.session.run(self.output, {self.input_states: [state]})[0]

    def optimise(self, memory):
        if len(memory.states) < 42:
            print('Memory is not big enough')
            return
        state_batch, actions, rewards = memory.get_batch(42)
        self.session.run(self.optimiser, feed_dict={self.input_states: state_batch, self.input_actions: actions, self.rewards: rewards})


class PolicyModel:
    def __init__(self):
        self.input_states = tf.placeholder(np.float32, shape=[None, 4], name='input')
        self.input_actions = tf.placeholder(np.float32, shape=[None, 2], name='q_values_new')
        self.rewards = tf.placeholder(np.float32, shape=[None], name='rewards')
        self.learning_rate = 1e-1

        net = self.input_states
        init = tf.random_normal_initializer(stddev=0.01)

        # hidden = tf.layers.dense(inputs=self.input_states, units=36, activation=tf.nn.relu, kernel_initializer=init, name='dense_1')
        # net = tf.layers.dense(inputs=hidden, units=36, activation=tf.nn.relu, kernel_initializer=init, name='dense_2')
        # logits = tf.layers.dense(inputs=hidden, units=2, activation=None, name='output')
        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self.input_states,
            num_outputs=36,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(stddev=1)
        )
        # hidden2 = tf.contrib.layers.fully_connected(
        #     inputs=hidden1,
        #     num_outputs=36,
        #     activation_fn=tf.nn.relu,
        #     weights_initializer=tf.random_normal_initializer
        # )

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=2,
            activation_fn=None
        )

        # op to sample an action
        self._sample = tf.reshape(tf.multinomial(logits, 1), [])

        # get log probabilities
        log_prob = tf.log(tf.nn.softmax(logits))

        # training part of graph
        self._acts = tf.placeholder(tf.int32)
        self._advantages = tf.placeholder(tf.float32)

        # get log probabilities of actions from episode
        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        # surrogate loss
        loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages))
        self._train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def run(self, observation):
        return self.session.run(self._sample, {self.input_states: [observation]})

    def train(self, obs, acts, ads):
        feed_dict = {self.input_states: obs, self._acts: acts, self._advantages: ads}
        self.session.run(self._train, feed_dict)


class ReplayMemory:
    def __init__(self, max_memory_size, gamma, model):
        self.states = deque(maxlen=max_memory_size)
        self.gamma = gamma
        self.model = model

    def add(self, state, actions, reward, done, next_state):
        self.states.append([state, actions, reward, done, next_state])

    # Here we get a batch of the states and the corresponding q values
    def get_batch(self, batch_size=32):
        mini_batch = random.sample(self.states, batch_size)
        states = [item[0] for item in mini_batch]
        actions = [item[1] for item in mini_batch]
        rewards = [item[2] for item in mini_batch]
        done = [item[3] for item in mini_batch]
        next_states = [item[4] for item in mini_batch]
        q_values = self.model.create_batch(next_states)

        y_batch = []

        for i in range(batch_size):
            if done[i]:
                y_batch.append(rewards[i])
            else:
                y_batch.append(rewards[i] + self.gamma * np.max(q_values[i]))

        return states, actions, y_batch


class Agent:

    def __init__(self):
        self.gamma = 0.99
        self.max_episodes = 400
        self.model = PolicyModel()
        self.render = False  # Should make it faster
        self.counts = []
        self.batches = 10

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, len(r))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    @staticmethod
    def process_rewards(rews):
        """Rewards -> Advantages for one episode. """

        # total reward: length of episode
        return [len(rews)] * len(rews)

    def run(self):
        env = gym.make('CartPole-v0')
        b_obs, b_acts, b_rews = [], [], []
        total_rewards = []
        for episode in range(self.max_episodes):
            current_state = env.reset()
            done = False
            count = 0
            obs, acts, rews = [], [], []
            while not done:

                if self.render:
                    env.render()

                action = self.model.run(current_state)

                next_state, reward, done, _ = env.step(action)  # observe the results from the action
                count += reward
                obs.append(current_state)
                acts.append(action)
                rews.append(reward)

                current_state = next_state

            total_rewards.append(len(rews))
            b_obs.extend(obs)
            b_acts.extend(acts)
            advantages = self.process_rewards(rews)
            b_rews.extend(advantages)

            if episode % self.batches == 0 and episode > 0:
                print('== TRAINING ==')
                # train
                b_rews = (b_rews - np.mean(b_rews)) // (np.std(b_rews) + 1e-10)
                self.model.train(b_obs, b_acts, b_rews)
                b_obs, b_acts, b_rews = [], [], []

            print('TRAIN: The episode ' + str(episode) + ' lasted for ' + str(count) + ' time steps')
            self.counts.append(count)

        # df = pd.Series(self.counts)
        # ma_counts = df.rolling(window=10).mean()
        # ma_counts = ma_counts.values
        # cleaned_list = [x for x in ma_counts if str(x) != 'nan']
        # cleaned_list = np.asarray(cleaned_list)
        # # 300 represents number of points to make between T.min and T.max
        # x_new = np.linspace(cleaned_list.min(), cleaned_list.max(), len(cleaned_list))
        # episodes = np.arange(0, len(cleaned_list))
        # power_smooth = spline(episodes, cleaned_list, x_new)

        plt.plot(total_rewards)
        plt.show()


agent = Agent()
agent.run()
