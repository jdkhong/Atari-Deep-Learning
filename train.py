import argparse
import tensorflow as tf
import numpy as np
import gym
import scipy.signal
from functools import partial
from typing import Iterable
import paac
import preatari

parser = argparse.ArgumentParser()
parser.add_argument("--t-max",
                    type=int,
                    default=50,
                    help="Update period")
parser.add_argument("--n-envs",
                    type=int,
                    default=4,
                    help="Number of parallel environments")

parser.add_argument("--logdir",
                    type=str,
                    default="plogdir",
                    help="Log directory")

parser.add_argument("--env",
                    type=str,
                    default="Pong-v0",
                    help="Environment Name")

FLAGS, _ = parser.parse_known_args()


def run_episodes(envs: Iterable[gym.Env], agent: paac.Agent, t_max=FLAGS.t_max, pipeline_fn=preatari.Preprocess.preprocess):
    """Summary
    Args:
        envs (Iterable[gym.Env]): A list of gym environments
        agent (Agent): Agent class
        t_max (int, optional): Max step to perform gradient updates
        pipeline_fn (function, optional): Preprocessing pipeline
    Returns:
        1-D Array: Reward array of shape (N_environments,)
    """
    n_envs = len(envs)
    all_dones = False

    states_memory = [[] for _ in range(n_envs)]
    actions_memory = [[] for _ in range(n_envs)]
    rewards_memory = [[] for _ in range(n_envs)]
    values_memory = [[] for _ in range(n_envs)]

    is_env_done = [False for _ in range(n_envs)]
    episode_rewards = [0 for _ in range(n_envs)]

    observations = []
    old_observations = []

    for id, env in enumerate(envs):
        s = env.reset()
        s = pipeline_fn(s)
        observations.append(s)
        old_observations.append(s)

    while not all_dones:

        for t in range(t_max):

            actions, values = agent.get_actions_values(observations)

            for id, env in enumerate(envs):

                if not is_env_done[id]:

                    s2, r, done, info = env.step(actions[id])

                    episode_rewards[id] += r
                    s2 = pipeline_fn(s2)
                    is_env_done[id] = done

                    states_memory[id].append(observations[id])
                    actions_memory[id].append(actions[id])
                    rewards_memory[id].append(r)
                    values_memory[id].append(values[id])

                    observations[id] = s2 - old_observations[id]
                    old_observations[id] = s2

                    if is_env_done[id]:
                        values_memory[id].append(0.0)

        future_values = agent.get_values(observations)

        for id in range(n_envs):
            if not is_env_done[id]:
                values_memory[id].append(future_values[id])

        agent.train(states_memory, actions_memory, rewards_memory, values_memory)

        states_memory = [[] for _ in range(n_envs)]
        actions_memory = [[] for _ in range(n_envs)]
        rewards_memory = [[] for _ in range(n_envs)]
        values_memory = [[] for _ in range(n_envs)]

        all_dones = np.all(is_env_done)

    return episode_rewards


def main():
    """MAIN """
    input_shape = [80, 80, 1]
    output_dim = 4
    pipeline_fn = partial(preatari.Preprocess.preprocess, new_HW=input_shape[:-1])

    envs = [gym.make(FLAGS.env) for i in range(FLAGS.n_envs)]
    envs[0] = gym.wrappers.Monitor(envs[0], "pmonitors", force=True)

    summary_writers = [tf.summary.FileWriter(logdir="{}/env-{}".format(FLAGS.logdir, i)) for i in range(FLAGS.n_envs)]
    agent = paac.Agent(input_shape, output_dim)

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)

    with tf.Session() as sess:
        try:
            if latest_checkpoint is not None:
                saver.restore(sess, latest_checkpoint)
                print("Restored from {}".format(latest_checkpoint))
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
                print("Initialized weights")

            episode = 1
            while True:
                rewards = run_episodes(envs, agent, pipeline_fn=pipeline_fn)
                print(episode, np.mean(rewards))
                print(rewards)
                print()

                for id, r in enumerate(rewards):
                    summary = tf.Summary()
                    summary.value.add(tag="Episode Reward", simple_value=r)
                    summary_writers[id].add_summary(summary, global_step=episode)
                    summary_writers[id].flush()

                if episode % 10 == 0:
                    saver.save(sess, "{}/model.ckpt".format(FLAGS.logdir), write_meta_graph=False)
                    print("Saved to {}/model.ckpt".format(FLAGS.logdir))

                episode += 1

        finally:
            saver.save(sess, "{}/model.ckpt".format(FLAGS.logdir), write_meta_graph=False)
            print("Saved to {}/model.ckpt".format(FLAGS.logdir))

            for env in envs:
                env.close()

            for writer in summary_writers:
                writer.close()


if __name__ == '__main__':
    main()
