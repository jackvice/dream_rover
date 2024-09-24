import numpy as np
import gym
import igibson
from igibson.envs.igibson_env import iGibsonEnv
import embodied

class DreamRover(embodied.Env):

    def __init__(self, config_file, mode='train', action_timestep=1.0/10.0, physics_timestep=1.0/240.0):
        self.env = iGibsonEnv(config_file=config_file, mode=mode, action_timestep=action_timestep, physics_timestep=physics_timestep)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._done = True

    @property
    def obs_space(self):
        spaces = {
            'depth': embodied.Space(np.float32, (64, 64, 1)),
            'scan': embodied.Space(np.float32, (228,)),
            'task_obs': embodied.Space(np.float32, (4,)),
            'proprioception': embodied.Space(np.float32, (12,)),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }
        return spaces

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.float32, (2,)),
            'reset': embodied.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self._done:
            obs = self.env.reset()
            self._done = False
            return self._obs(obs, 0.0, is_first=True)
        
        obs, reward, done, info = self.env.step(action['action'])
        self._done = done
        return self._obs(obs, reward, is_last=done)

    def _obs(self, obs, reward, is_first=False, is_last=False):
        return {
            'depth': obs['depth'],
            'scan': obs['scan'],
            'task_obs': obs['task_obs'],
            'proprioception': obs['proprioception'],
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_last,
        }

    def close(self):
        self.env.close()

def make(config, mode):
    config_file = 'path/to/your/igibson_config.yaml'  # Update this path
    return DreamRover(config_file, mode)

