import gym
from gym.wrappers import Monitor
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()

env = Monitor(gym.make('CartPole-v0'), './video', force=True)
env.reset()

done = False

while not done:
    env.render()
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)

env.close()
display.stop()
