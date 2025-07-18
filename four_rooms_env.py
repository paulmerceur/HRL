import gymnasium
from minigrid.wrappers import FullyObsWrapper
from gymnasium import spaces


class DictToBoxWrapper(gymnasium.ObservationWrapper):
    """A standard wrapper to extract the 'image' observation from the dict."""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict)
        self.observation_space = env.observation_space['image']

    def observation(self, obs):
        return obs['image']

def make_env(render_mode=None, max_episode_steps=512, **kwargs):
    """
    A standard environment factory function.
    This function creates and returns a basic, wrapped Gymnasium environment.
    """
    # Create the base environment.
    env = gymnasium.make(
        "MiniGrid-FourRooms-v0",
        render_mode=render_mode,
    )
    env.unwrapped.max_steps = max_episode_steps # type: ignore
    
    # Apply the observation wrappers
    env = FullyObsWrapper(env)
    env = DictToBoxWrapper(env)
    
    return env 

def make_env_for_vectorization(render_mode=None):
    """Create environment for standard gymnasium vectorization."""
    return make_env(render_mode=render_mode, max_episode_steps=512) 