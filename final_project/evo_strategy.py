# Libraries and Constants
import multiprocessing as mp

import gym
from gym.wrappers import Monitor
from pyvirtualdisplay import Display

import numpy as np
from numpy.random import default_rng as rng

POP_SIZE = 20
N_GEN = 1100
SIGMA = 0.05
LR = 0.05
N_WORKERS = mp.cpu_count() - 1
CONFIG = {
    'game': 'Pendulum-v0',
    'input_dim': 3,
    'output_dim': 1,
    'cont_action': [True, 2.0],
    'max_steps': 200,
    'fitness': -180
}
LAYER_DIMS = [
    (CONFIG['input_dim'], 30),
    (30, 20),
    (20, CONFIG['output_dim'])
]
PARAM_SIZE = np.sum([n_w * n_b + n_b for n_w, n_b in LAYER_DIMS])


# Helper Functions
def get_epsilon(seed, m_id):
    sign = -1.0 if m_id % 2 == 0 else 1.0
    return sign * rng(seed).standard_normal(PARAM_SIZE)


def flat_linear(n_in, n_out):
    w = rng().standard_normal(n_in * n_out) * 0.1
    b = rng().standard_normal(n_out) * 0.1
    return np.concatenate([w, b])


# Optimizer
class Optimizer:
    def __init__(self, momentum=None):
        rank = np.arange(1, POP_SIZE + 1)
        c = np.maximum(0.0, np.log(POP_SIZE // 2 + 1) - np.log(rank))
        self.util = (c / c.sum()) - (1 / POP_SIZE)
        self.m = momentum
        if momentum is not None:
            self.grad = np.zeros(PARAM_SIZE)

    def get_gradient(self, seeds, rewards):
        rank = np.argsort(rewards)[::-1]
        gradient = np.zeros(PARAM_SIZE)

        for u_id, m_id in enumerate(rank):
            epsilon = get_epsilon(seeds[m_id], m_id)
            gradient += self.util[u_id] * epsilon
        gradient /= POP_SIZE * SIGMA

        if self.m is not None:
            self.grad = self.m * self.grad + (1 - self.m) * gradient
            return self.grad
        else:
            return gradient


# Evaluate Mutant
class Net:
    def __init__(self, params):
        self.layers, start = [], 0
        for n_in, n_out in LAYER_DIMS:
            n_w, n_b = n_in * n_out, n_out
            self.layers.append([
                params[start: start + n_w].reshape((n_in, n_out)),
                params[start + n_w: start + n_w + n_b].reshape((1, n_out))
            ])
            start += n_w + n_b

    def get_action(self, x):
        for w, b in self.layers:
            x = np.tanh(x @ w + b)

        if not CONFIG['cont_action'][0]:
            return np.argmax(x, axis=1)[0]
        else:
            return CONFIG['cont_action'][1] * np.tanh(x)[0]


def eval_mutant(env, parent, m_id=None, seed=None):
    if m_id is not None and seed is not None:
        epsilon = get_epsilon(seed, m_id)
        mutant = parent + SIGMA * epsilon
        net = Net(mutant)
    else:
        net = Net(parent)

    episode_reward = 0.0
    state = env.reset()
    for _ in range(CONFIG['max_steps']):
        action = net.get_action(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break

    return episode_reward


# Evolve
def evolve(env, genotype, opt, pool):
    seeds = rng().integers(0, 2 ** 32 - 1, size=POP_SIZE//2).repeat(2)

    jobs = [
        pool.apply_async(eval_mutant, (env, genotype, m_id, seeds[m_id]))
        for m_id in range(POP_SIZE)
    ]
    rewards = np.array([j.get() for j in jobs])

    grad = opt.get_gradient(seeds, rewards)
    genotype += LR * grad


# Visualize results
def visualize(env, genotype):
    print('Testing....\n')
    display = Display(visible=0, size=(1400, 900))
    env = Monitor(env, './video', force=True)
    net = Net(genotype)

    display.start()
    state = env.reset()

    for _ in range(CONFIG['ep_max_steps']):
        env.render()
        action = net.get_action(state)
        state, _, done, _ = env.step(action)
        if done:
            break
    else:
        env.stats_recorder.save_complete()
        env.stats_recorder.done = True

    env.close()
    display.stop()


# Main Function
if __name__ == '__main__':
    env = gym.make(CONFIG['game']).unwrapped
    genotype = np.concatenate([
        flat_linear(n_in, n_out) for n_in, n_out in LAYER_DIMS
    ])
    opt = Optimizer(momentum=0.9)
    pool = mp.Pool(processes=N_WORKERS)
    mar = None

    for t in range(N_GEN):
        evolve(env, genotype, opt, pool)
        rewards = eval_mutant(env, genotype)

        mar = rewards if mar is None else 0.9 * mar + 0.1 * rewards
        print(f'{t}, {mar:.3f}')

        if mar > CONFIG['fitness']:
            break
