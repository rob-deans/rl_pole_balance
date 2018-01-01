import gym
from collections import deque
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.input_states = tf.placeholder(np.float32, shape=[None, 4], name='input')
        self.input_actions = tf.placeholder(np.float32, shape=[None, 2], name='q_values_new')
        self.rewards = tf.placeholder(np.float32, shape=[None], name='rewards')
        self.learning_rate = 1e-4

        net = self.input_states
        init = tf.truncated_normal_initializer()

        net = tf.layers.dense(inputs=net, units=10, activation=tf.nn.relu, kernel_initializer=init, name='dense_1')
        net = tf.layers.dense(inputs=net, units=20, activation=tf.nn.relu, kernel_initializer=init, name='dense_2')
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
        if len(memory.states) < 32:
            print('Memory is not big enough')
            return
        state_batch, actions, rewards = memory.get_batch(32)
        self.session.run(self.optimiser, feed_dict={self.input_states: state_batch, self.input_actions: actions, self.rewards: rewards})


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
        self.gamma = .97
        self.max_episodes = 1000
        self.epsilon = 1.
        self.final_epsilon = .1
        self.model = Model()
        self.memory = ReplayMemory(max_memory_size=10000, gamma=self.gamma, model=self.model)
        self.render = False  # Should make it faster
        self.counts = []

    def run(self):
        env = gym.make('CartPole-v0')
        for i_episode in range(self.max_episodes):
            current_state = env.reset()
            done = False
            count = 0
            while not done:

                if self.render:
                    env.render()

                actions = np.zeros(2)
                if random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = self.model.predict(current_state)
                    action = np.argmax(q_values)
                actions[action] = 1

                next_state, reward, done, _ = env.step(action)  # observe the results from the action
                count += 1

                self.memory.add(current_state, actions, reward, done, next_state)

                current_state = next_state

                self.model.optimise(self.memory)
            print('TRAIN: The episode ' + str(i_episode) + ' lasted for ' + str(count) + ' timesteps with epsilon ' + str(self.epsilon))
            self.counts.append(count)

            if i_episode % 100 == 0 and i_episode > 1:  # TODO: epsilon annealing?
                if self.epsilon > self.final_epsilon:
                    self.epsilon -= self.final_epsilon

        plt.plot(self.counts)
        plt.show()


agent = Agent()
agent.run()
