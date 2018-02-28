"""Glues together an experiment, agent, and environment.
"""

from __future__ import print_function

import math
import time
from abc import ABCMeta, abstractmethod


def timing(f):
    stats = {'mean': 0,
             'M2': 0,
             'n': 0}
    msg = "Loop took {:.4f}/{:.4f}/{:.2f}% current/mean/rsd seconds."

    def wrap(*args, **kwargs):
        start = time.time()
        out = f(*args, **kwargs)
        end = time.time()
        x = end - start
        old_mean = stats['mean']
        stats['n'] += 1
        stats['mean'] += (x - stats['mean']) / stats['n']
        stats['M2'] += (x - old_mean) * (x - stats['mean'])

        rsd = 100 * math.sqrt(stats['M2'] / stats['n']) / abs(stats['mean'])
        print(msg.format(x, stats['mean'], rsd))

        return out

    return wrap


class RLGlue:
    """RLGlue class.

    args:
        env_obj: an object that implements BaseEnvironment
        agent_obj: an object that implements AgentEnvironment
    """

    def __init__(self, env_obj, agent_obj):
        self.environment = env_obj
        self.agent = agent_obj

        self.total_reward = 0
        self.last_action = 0
        self.num_steps = 0
        self.num_episodes = 0
        self.num_ep_steps = 0

    def rl_start(self):
        """Starts RLGlue experiment.

        Returns:
            tuple: (state, action)
        """
        self.num_ep_steps = 0
        self.total_reward = 0

        last_obs = self.environment.env_start()['obs']
        self.last_action = self.agent.agent_start(last_obs)

        return {'obs': last_obs,
                'action': self.last_action}

    def rl_env_start(self):
        """Useful when manually specifying agent actions (for debugging). Starts
        RL-Glue environment.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination
        """
        self.total_reward = 0.0
        self.num_ep_steps = 1

        return self.environment.env_start()

    def rl_env_step(self, action):
        """Useful when manually specifying agent actions (for debugging).Takes a
        step in the environment based on an action.

        Args:
            action: Action taken by agent.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        env_dict = self.environment.env_step(action)

        self.total_reward += env_dict['reward']

        if env_dict['terminal']:
            self.num_episodes += 1
        else:
            self.num_ep_steps += 1

        self.num_steps += 1

        return env_dict

    @timing
    def rl_step(self):
        """Takes a step in the RLGlue experiment.

        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """
        env_dict = self.environment.env_step(self.last_action)
        reward = env_dict['reward']

        self.total_reward += reward

        if env_dict['terminal']:
            self.num_episodes += 1
            self.agent.agent_end(reward)
            env_dict['action'] = None
        else:
            self.last_action = self.agent.agent_step(reward, env_dict['obs'])
            env_dict['action'] = self.last_action

        self.num_ep_steps += 1
        self.num_steps += 1

        return env_dict

    def rl_episode(self, max_steps_this_episode):
        """Convenience function to run an episode.

        Args:
            max_steps_this_episode (Int): Max number of steps in this episode.
                A value of 0 will result in the episode running until
                completion.
        """
        is_terminal = False

        self.rl_start()

        while (not is_terminal and
               ((max_steps_this_episode == 0) or
                (self.num_ep_steps < max_steps_this_episode))):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result['terminal']


class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        __init__, agent_start, agent_step, and agent_end are required methods.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Initialize agent variables."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """


class BaseEnvironment:
    """Implements the environment for an RLGlue environment

    Note:
        __init__, env_start, and env_step are required methods.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Initialize environment variables."""
        reward = None
        observation = None
        terminal = None
        env_dict = {'reward': reward,
                    'obs': observation,
                    'terminal': terminal}

    @abstractmethod
    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

    @abstractmethod
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
