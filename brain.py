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

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self.input_states,
            num_outputs=36,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer
        )
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


class ActorCriticModel:
    def __init__(self, memory, env):
        # General
        self.memory = memory
        self.discount_factor = 0.99

        # Environment
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.value_size = 1

        self.actor_lr = 1e-3
        self.critic_lr = 5e-3

        # ============================ #
        #            Actor             #
        # ============================ #
        self.actor_input_state = tf.placeholder(tf.float32, shape=[None, self.state_space], name='actor_i_state')
        self.actor_input_action = tf.placeholder(tf.float32, shape=[None, self.action_space], name='actor_i_act')
        self.actor_td_error = tf.placeholder(tf.float32, shape=[None, 1], name='td_placeholder')

        # init = tf.truncated_normal_initializer(0, 0.01)
        init = tf.uniform_unit_scaling_initializer

        net = tf.layers.dense(inputs=self.actor_input_state, units=36, activation=tf.nn.relu, kernel_initializer=init, name='dense_1')
        self.actor_output = tf.layers.dense(inputs=net, units=self.action_space, kernel_initializer=init, activation=tf.nn.softmax, name='output')

        # Categorical cross entropy
        loss = tf.log(tf.reduce_sum(tf.multiply(self.actor_input_action, self.actor_output))) * self.actor_td_error
        self.actor_optimise = tf.train.AdamOptimizer(self.actor_lr).minimize(-loss)

        # ============================ #
        #            Critic            #
        # ============================ #

        self.critic_input_state = tf.placeholder(tf.float32, shape=[None, self.state_space])
        self.critic_td_target = tf.placeholder(tf.float32, shape=[None, 1], name='critic_td')

        critic_net = self.critic_input_state

        critic_net = tf.layers.dense(inputs=critic_net, units=100, activation=tf.nn.relu, kernel_initializer=init)
        critic_net = tf.layers.dense(inputs=critic_net, units=150, activation=tf.nn.relu, kernel_initializer=init)
        critic_net = tf.layers.dense(inputs=critic_net, units=100, activation=tf.nn.relu, kernel_initializer=init)

        self.critic_output = tf.layers.dense(inputs=critic_net, units=self.value_size, activation=None, kernel_initializer=init)

        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.critic_td_target, self.critic_output))
        self.critic_optimise = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

        # Tensorflow session init
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def run(self, state):
        actions = self.session.run(self.actor_output, feed_dict={self.actor_input_state: [state]})[0]
        return np.random.choice(self.action_space, 1, p=actions)[0]

    def predict(self, state):
        return self.session.run(self.critic_output, feed_dict={self.critic_input_state: [state]})[0]

    def batch_predict(self, states):
        return self.session.run(self.critic_output, feed_dict={self.critic_input_state: states})

    def train(self):
        if len(self.memory.mem) < self.memory.batch_size:
            return
        samples = self.memory.get()

        td_targets = []
        td_errors = []

        states = [sample[0] for sample in samples]
        actions = [sample[1] for sample in samples]
        rewards = [sample[2] for sample in samples]
        done = [sample[3] for sample in samples]
        next_states = [sample[4] for sample in samples]

        values = self.batch_predict(states)
        value_primes = self.batch_predict(next_states)

        for i in range(self.memory.batch_size):
            if done[i]:
                td_targets.append([rewards[i]])
            else:
                td_targets.append(rewards[i] + self.discount_factor * value_primes[i])

            td_errors.append(td_targets[-1] - values[i])

        self.session.run(self.critic_optimise, feed_dict={self.critic_input_state: states,
                                                          self.critic_td_target: td_targets})

        self.session.run(self.actor_optimise, feed_dict={self.actor_input_state: states,
                                                         self.actor_input_action: actions,
                                                         self.actor_td_error: td_errors})


class Memory:
    def __init__(self):
        self.mem = deque(maxlen=2000)
        self.batch_size = 32

    def add(self, state, action, reward, done, next_state):
        self.mem.append([state, action, reward, done, next_state])

    def get(self):
        return random.sample(self.mem, self.batch_size)


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
        self.max_episodes = 1000
        self.gamma = 0.99
        self.env = gym.make('CartPole-v0')
        self.model = ActorCriticModel(Memory(), env=self.env)
        self.render = False
        self.counts = []

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
        for episode in range(self.max_episodes):
            current_state = self.env.reset()
            done = False
            count = 0
            while not done:

                if self.render:
                    self.env.render()

                action = self.model.run(current_state)

                next_state, reward, done, _ = self.env.step(action)  # observe the results from the action
                count += reward

                if done and count < 200:
                    reward = -100
                actions = np.zeros(self.model.action_space)
                actions[action] = 1
                self.model.memory.add(current_state, actions, reward, done, next_state)

                self.model.train()
                current_state = next_state

            print('TRAIN: The episode ' + str(episode) + ' lasted for ' + str(count) + ' time steps')
            self.counts.append(count)

        plt.plot(self.counts)
        plt.show()


agent = Agent()
agent.run()
