import embodied
import numpy as np


class Acrobot(embodied.Env):

  def __init__(self, task, size=(64, 64), length=100):
    assert task in ('cont', 'disc')
    self._task = task
    self._size = size
    self._length = length
    self._step = 0
    self._done = False

  @property
  def obs_space(self):
    return {
        'vector': embodied.Space(np.float32, (6,)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, 2),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      return self._obs(0, is_first=True)
    action = action['action']
    self._step += 1
    self._done = (self._step >= self._length)
    return self._obs(1, is_last=self._done, is_terminal=self._done)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    return dict(
        image=np.zeros(self._size + (3,), np.uint8),
        vector=np.zeros(7, np.float32),
        token=np.zeros((), np.int32),
        step=np.float32(self._step),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
