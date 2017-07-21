import argparse
import tensorflow as tf
import numpy as np
import gym
import scipy.signal
from functools import partial
from typing import Iterable


parser = argparse.ArgumentParser()
parser.add_argument("--epsilon",
                    type=float,
                    default=1e-7,
                    help="Epsilon value for numeric stability")

parser.add_argument("--decay",
                    type=float,
                    default=.99,
                    help="Decay rate for RMSProp and Discount Rate")

parser.add_argument("--lambda",
                    type=float,
                    dest="lambda_",
                    default=.99,
                    help="Decay rate lambda for Generalized Advantage Estimation")

parser.add_argument("--learning-rate",
                    type=float,
                    default=0.001,
                    help="Learning rate for RMSProp")


parser.add_argument("--entropy",
                    type=float,
                    default=0.01,
                    help="Entropy coefficient")


parser.add_argument("--logdir",
                    type=str,
                    default="plogdir",
                    help="Log directory")


FLAGS, _ = parser.parse_known_args()


class Agent(object):

    def __init__(self, input_shape: list, output_dim: int):
        """Main Agent files
        Args:
            input_shape (list): Input shape [H, W, C]
            output_dim (int): Number of actions
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.__build_network(self.input_shape, self.output_dim)

    def discount(x, gamma):
        """Discount rewards by a rate of `gamma`
        Args:
            x (1-D Array): Rewards array of shape (N,)
            gamma (float): Discount rate
        Returns:
            1-D Array: Discounted rewards
        """
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def __build_network(self, input_shape: list, output_dim: int):
        """Build a network """
        self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
        self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
        action_onehots = tf.one_hot(self.actions, depth=output_dim, name="action_onehots")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.advantages = tf.placeholder(tf.float32, shape=[None], name="advantages")

        net = self.states
        with tf.variable_scope("layer1"):
            net = tf.layers.conv2d(net, filters=16, kernel_size=(8, 8), strides=(4, 4), name="conv")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("layer2"):
            net = tf.layers.conv2d(net, filters=32, kernel_size=(4, 4), strides=(2, 2), name="conv")
            net = tf.nn.relu(net, name="relu")

        net = tf.contrib.layers.flatten(net)

        with tf.variable_scope("fc1"):
            net = tf.layers.dense(net, units=256, name="fc")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("action_network"):
            action_scores = tf.layers.dense(net, units=output_dim, name="action_scores")
            self.action_probs = tf.nn.softmax(action_scores, name="action_probs")
            single_action_prob = tf.reduce_sum(self.action_probs * action_onehots, axis=1)
            log_action_prob = - tf.log(tf.clip_by_value(single_action_prob, FLAGS.epsilon, 1.0)) * self.advantages
            action_loss = tf.reduce_sum(log_action_prob)

        with tf.variable_scope("entropy"):
            entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs + FLAGS.epsilon), axis=1)
            entropy_sum = tf.reduce_sum(entropy)

        with tf.variable_scope("value_network"):
            self.values = tf.squeeze(tf.layers.dense(net, units=1, name="values"))
            value_loss = tf.reduce_sum(tf.squared_difference(self.rewards, self.values))

        with tf.variable_scope("total_loss"):
            self.loss = action_loss + value_loss * 0.5 - entropy_sum * FLAGS.entropy

        with tf.variable_scope("train_op"):
            self.optim = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate,
                                                   decay=FLAGS.decay)
            self.train_op = self.optim.minimize(self.loss, global_step=tf.contrib.framework.get_or_create_global_step())

        tf.summary.histogram("Action Probs", self.action_probs)
        tf.summary.histogram("Entropy", entropy)
        tf.summary.histogram("Actions", self.actions)
        tf.summary.scalar("Loss/total", self.loss)
        tf.summary.scalar("Loss/value", value_loss)
        tf.summary.scalar("Loss/actor", action_loss)
        tf.summary.image("Screen", self.states[:, :, :, -1:])

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter("{}/main".format(FLAGS.logdir), graph=tf.get_default_graph())

    def get_actions(self, states):
        """
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)
        Returns:
            actions (1-D Array): Action Array of shape (N,)
        """
        sess = tf.get_default_session()
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        action_probs = sess.run(self.action_probs, feed)
        noises = np.random.uniform(size=action_probs.shape[0]).reshape(-1, 1)

        return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1)

    def get_values(self, states):
        """
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)
        Returns:
            values (1-D Array): Values (N,)
        """
        sess = tf.get_default_session()
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        return sess.run(self.values, feed).reshape(-1)

    def get_actions_values(self, states):
        """Returns actions and values
        Args:
            states (4-D Array): State Array of shape (N, H, W, C)
        Returns:
            actions (1-D Array): actions of shape (N,) with int dtype
            values (1-D Array): values of shape (N,) with int dtype
        """
        sess = tf.get_default_session()
        feed = {
            self.states: states
        }
        action_probs, values = sess.run([self.action_probs, self.values], feed)
        noises = np.random.uniform(size=action_probs.shape[0]).reshape(-1, 1)

        return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1), values.reshape(-1)

    def train(self, states, actions, rewards, values):
        """Update parameters by gradient descent
        Args:
            states (5-D Array): Image arrays of shape (n_envs, n_timesteps, H, W, C)
            actions (2-D Array): Action arrays of shape (n_envs, n_timesteps)
            rewards (2-D Array): Rewards array of shape (n_envs, n_timesteps)
            values (2-D Array): Value array of shape (n_envs, n_timesteps)
        """
        n_envs = len(states)

        states = np.vstack([s for s in states if len(s) > 0])
        actions = np.hstack(actions)

        discounted_rewards = []
        advantages = []

        for id in range(n_envs):
            if len(rewards[id]) > 0:
                rewards_bootstrap = rewards[id] + [values[id][-1]]
                discounted_rewards.extend(Agent.discount(rewards_bootstrap, FLAGS.decay)[:-1])

                assert len(rewards[id]) < len(values[id]), "{}, {}".format(np.array(rewards[id]).shape, np.array(values[id]).shape)
                delta_t = np.array(rewards[id]) + FLAGS.decay * np.array(values[id][1:]) - np.array(values[id][:-1])
                advantages.extend(Agent.discount(delta_t, FLAGS.decay * FLAGS.lambda_))

        sess = tf.get_default_session()
        feed = {
            self.states: states,
            self.actions: actions,
            self.rewards: discounted_rewards,
            self.advantages: advantages
        }
        _, summary_op, global_step = sess.run([self.train_op,
                                               self.summary_op,
                                               tf.train.get_global_step()],
                                              feed_dict=feed)
        self.summary_writer.add_summary(summary_op, global_step=global_step)
