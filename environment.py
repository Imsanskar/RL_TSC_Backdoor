import gym
import numpy as np


class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.
    Parameters
    ----------
    world: World object
    agents: list of agents, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """

    def __init__(self, world, agents, metric, attacker_agents = None):
        """
        :param world: one world object to interact with agents. Support multi world
        objects in different TSCEnvs.
        :param agents: single agents, each control all intersections. Or multi agents,
        each control one intersection.
        actions is a list of actions, agents is a list of agents.
        :param metric: metrics to evaluate policy.
        """
        self.world = world
        self.eng = self.world.eng
        self.n_agents = len(agents) * agents[0].sub_agents
        # test agents number == intersection number
        assert len(world.intersection_ids) == self.n_agents
        self.agents = agents
        action_dims = [agent.action_space.n * agent.sub_agents for agent in agents]
        # total action space of all agents.
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        self.metric = metric
        self.attacker_agents = attacker_agents
        self.previous_metric_value = 0
    def step(self, actions):
        """
        Execute one environment step with victim agent actions.

        :param actions: Victim agent actions (signal phase choices), shape: (N_agents,)
        :return: (obs, rewards, dones, infos) - observations after stepping

        [NORMAL VICTIM STEP]:
        1. Advance simulation with victim signal actions
        2. Query each victim agent for poisoned/unpoisoned observation
        3. Compute rewards from traffic metrics
        """
        if not actions.shape:
            assert(self.n_agents == 1)
            actions = actions[np.newaxis]
        else:
            assert len(actions) == self.n_agents

        # Advance simulation with victim signal actions
        self.world.step(actions)

                # Get observations from all agents (includes fake vehicles if injected)
        if self.attacker_agents is not None and len(self.attacker_agents) > 0:
            # get reward for the attacker agents based on the current state of the environment after stepping
            # TODO: What could be the best reward function for the attacker? Maybe a combination of delay increase and stealthiness?
            lane_waiting_vehicles = self.world.get_info("lane_waiting_count")
            attacker_reward = sum(lane_waiting_vehicles.values()) - self.previous_metric_value
            # self.previous_metric_value = sum(lane_waiting_vehicles.values())
            obs = [agent.get_ob() for agent in self.agents]
            rewards = [attacker_reward]
        else:
            if not len(self.agents) == 1:
                obs = [agent.get_ob() for agent in self.agents]
                rewards = [agent.get_reward() for agent in self.agents]
            else:
                obs = [self.agents[0].get_ob()]
                rewards = [self.agents[0].get_reward()]

        dones = [False] * self.n_agents
        infos = {}

        return obs, rewards, dones, infos



    def reset(self):
        self.world.reset()

        obs = None
        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]  # [agent, sub_agent==1, feature]
        else:
            obs = [self.agents[0].get_ob()]  # [agent==1, sub_agent, feature]
        return obs
