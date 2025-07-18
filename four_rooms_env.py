import gymnasium
from minigrid.wrappers import FullyObsWrapper
from gymnasium import spaces


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
    
    # Apply the observation wrappers - only FullyObsWrapper to keep Dict obs
    env = FullyObsWrapper(env)
    # env = DictToBoxWrapper(env)  # Removed to keep full Dict observation
    
    return env 