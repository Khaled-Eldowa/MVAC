from gym.envs.registration import registry, register, make, spec

register(
    id='PointReacher-v4',
    entry_point='envs.point_v4:PointReacher',
    max_episode_steps=None,
    kwargs={'terminating': False},
    reward_threshold=None,
)
