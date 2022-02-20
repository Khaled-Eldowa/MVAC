import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math as m


class PointReacher(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, terminating=False):
        self.max_pos = 10.0
        self.max_action = 2.0
        self.penalty_thresh = 1.0
        self.min_noise_var = 0.01
        self.goal_radius = 0.05
        self.terminating = terminating

        # gym attributes
        high = np.array([self.max_pos])
        self.observation_space = spaces.Box(low=np.float32(-high), high=np.float32(high))
        self.action_space = spaces.Box(low=np.float32(-self.max_action), high=np.float32(self.max_action), shape=(1,))

        self.viewer = None
        self.state = None
        self.seed()

    def step(self, action, render=False):

        action = np.clip(action, -self.max_action, self.max_action)
        noise_var = (np.abs(action) + self.min_noise_var) ** (1 / 4)
        new_state = self.state + action + self.np_random.randn() * np.sqrt(noise_var)
        new_state = np.clip(new_state, -self.max_pos, self.max_pos)

        action_penalty = np.abs(action) ** 3

        reward = -np.asscalar((np.abs(new_state) / self.max_pos) ** (1 / 4) +
                              self.np_random.randn() * np.sqrt(action_penalty)*0.25)

        done = np.abs(new_state) < self.goal_radius and self.terminating

        self.state = new_state

        return self._get_state(), reward, done, {}

    def reset(self, state=None):
        if state is None:
            init_pos = self.np_random.uniform(low=self.max_pos / 2 - 0.1, high=self.max_pos / 2)
            sign = self.np_random.choice([-1, 1])
            self.state = np.array([init_pos * sign])
        else:
            self.state = np.array(state)

        return self._get_state()

    def _get_state(self):
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = (self.max_pos * 2) * 2
        scale = screen_width / world_width
        bally = 100
        ballradius = 3

        if self.viewer is None:
            clearance = 0  # y-offset
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, clearance)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            self.track = rendering.Line((0, bally), (screen_width, bally))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            zero_line = rendering.Line((screen_width / 2, 0),
                                       (screen_width / 2, screen_height))
            zero_line.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line)

        x = self.state[0]
        ballx = x * scale + screen_width / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
