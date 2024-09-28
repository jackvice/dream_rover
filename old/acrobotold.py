import gym
import numpy as np

class Acrobot:
    def __init__(self, action_repeat=1):
        self._env = gym.make('Acrobot-v1')
        self._action_repeat = action_repeat

    @property
    def obs_space(self):
        spaces = {
            'state': self._env.observation_space,
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
            'log_reward': embodied.Space(np.float32),
        }
    if self._logs:
      spaces.update({
          f'log_achievement_{k}': embodied.Space(np.int32)
          for k in self._achievements})
    return spaces

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._action_repeat):
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            if done:
                break
        
        obs_dict = {
            'state': obs,
            'is_first': False,
            'is_terminal': done,
        }
        return obs_dict, total_reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs_dict = {
            'state': obs,
            'is_first': True,
            'is_terminal': False,
        }
        return obs_dict

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()

