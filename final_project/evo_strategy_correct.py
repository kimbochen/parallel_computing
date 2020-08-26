import multiprocessing as mp
from time import time

import gym
from gym.wrappers import Monitor
from pyvirtualdisplay import Display

import numpy as np
from numpy.random import default_rng as rng

POP_SIZE = 20
N_GEN = 1200
LR = 0.05
SIGMA = 0.05
N_WORKERS = mp.cpu_count() - 1
CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180)
][2]
LAYER_DIMS = [
    (CONFIG['n_feature'], 30),
    (30, 20),
    (20, CONFIG['n_action'])
]
PARAM_SIZE = np.sum([n_w * n_b + n_b for n_w, n_b in LAYER_DIMS])


class SGD:
    def __init__(self, momentum=0.9):
        util = np.maximum(
            0, np.log(POP_SIZE // 2 + 1) - np.log(np.arange(1, POP_SIZE + 1))
        )

        self.utility = util / util.sum() - 1 / POP_SIZE
        self.v = np.zeros(PARAM_SIZE).astype(np.float32)
        self.momentum = momentum

    def gradients(self, rewards, noise_seed):
        kids_rank = np.argsort(rewards)[::-1]

        u = np.array([self.utility[i] for i in range(kids_rank.size)])
        e = np.array([epsilon(noise_seed[k_id], k_id) for k_id in kids_rank])

        grad = u @ e / (POP_SIZE * SIGMA)
        self.v = self.momentum * self.v + (1.0 - self.momentum) * grad

        return self.v


def epsilon(seed, m_id):
    sign = -1.0 if m_id % 2 == 0 else 1.0
    return sign * rng(seed).standard_normal(PARAM_SIZE)

def flat_linear(n_in, n_out):
    w = rng().standard_normal(n_in * n_out) * 0.1
    b = rng().standard_normal(n_out) * 0.1
    return np.concatenate([w, b])

def create_nn(params):
    nn, start = [], 0
    for n_in, n_out in LAYER_DIMS:
        n_w, n_b = n_in * n_out, n_out
        nn.append([
            params[start : start+n_w].reshape([n_in, n_out]),
            params[start+n_w : start+n_w+n_b].reshape([1, n_out])
        ])
        start += n_w + n_b
    return nn

def get_action(nn, x):
    for w, b in nn[0:-1]:
        x = np.tanh(x @ w + b)
    w, b = nn[-1]
    x = x @ w + b
    
    if not CONFIG['continuous_a'][0]:
        return np.argmax(x, axis=1)[0]
    else:
        return CONFIG['continuous_a'][1] * np.tanh(x)[0]


def get_reward(params, env, seed_and_id=None):
    if seed_and_id is not None:
        params += SIGMA * epsilon(*seed_and_id)

    nn = create_nn(params)
    state = env.reset()
    episode_rewards = 0.0

    for step in range(CONFIG['ep_max_step']):
        action = get_action(nn, state)
        state, reward, done, _ = env.step(action)
        episode_rewards += reward
        if done: break

    return episode_rewards

def train(net_params, optimizer, pool):
    noise_seed = rng().integers(0, 2**32-1, size=POP_SIZE//2).repeat(2)
    jobs = [pool.apply_async(
        get_reward, (net_params, env, [noise_seed[k_id], k_id]))
        for k_id in range(POP_SIZE)
    ]
    rewards = np.array([j.get() for j in jobs])
    gradients = optimizer.gradients(rewards, noise_seed)

    return net_params + LR * gradients, rewards

def visualize(env, net_params):
    print('Testing....\n')
    display = Display(visible=0, size=(1400, 900))
    env = Monitor(env, './video', force=True)
    nn = create_nn(net_params)

    display.start()
    state = env.reset()

    for _ in range(CONFIG['ep_max_steps']):
        env.render()
        action = get_action(nn, state)
        state, _, done, _ = env.step(action)
        if done: break
    else:
        env.stats_recorder.save_complete()
        env.stats_recorder.done = True

    env.close()
    display.stop()

if __name__ == "__main__":
    env = gym.make(CONFIG['game']).unwrapped
    net_params = np.concatenate([
        flat_linear(n_in, n_out) for n_in, n_out in LAYER_DIMS
    ])
    optimizer = SGD()
    pool = mp.Pool(processes=N_WORKERS)
    mar = None  # moving average reward

    for g in range(N_GEN):
        t0 = time()
        net_params, kid_rewards = train(net_params, optimizer, pool)

        net_r = get_reward(net_params, env)
        mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r
        print(
            f'Gen: {g} | Net_R: {mar:.1f} | '
            f'Kid_avg_R: {kid_rewards.mean():.1f} | '
            f'Gen_T: {(time() - t0):.2f}'
        )

        if mar > CONFIG['eval_threshold']: break

    visualize(env, net_params)
