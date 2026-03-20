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
        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]
            rewards = [agent.get_reward() for agent in self.agents]
        else:
            obs = [self.agents[0].get_ob()]
            rewards = [self.agents[0].get_reward()]

        dones = [False] * self.n_agents
        infos = {}

        return obs, rewards, dones, infos

    def attack_step(self, attacker_agents=None, victim_actions=None):
        """
        Execute one adversarial attack step following the Multi-PPO strategy.

        [ATTACK FLOW FROM WIP PAPER]:
        1. Attacker observes current traffic state from victim environment
        2. Attacker's PPO policy selects attack action (approach + vehicle counts)
        3. Attacker injects fake vehicles via SDSM injector into simulation
        4. Environment state is modified to include fake vehicles (poisoned)
        5. Victim agent queries environment → gets poisoned observation (includes fake CAVs)
        6. Victim agent predicts signal phase based on poisoned observation
        7. Environment advances with victim's phase decision
        8. Compute attacker reward: maximize delay increase + stealth bonus

        Args:
            attacker_agents: List of MultiPPOAttacker instances (one per intersection)
            victim_actions: Optional victim actions to use for environment stepping
                           If None, uses default/stay action

        Returns:
            Tuple of (attacker_states, victim_observations, attacker_rewards, infos)
        """
        if attacker_agents is None or len(attacker_agents) == 0:
            # No attackers active - just do normal step
            return self.step(np.zeros(self.n_agents, dtype=np.int64))

        if not hasattr(attacker_actions, '__len__') and victim_actions is not None:
            attacker_actions = [attacker_actions]

        # === Step 1: Each attacker observes state and gets attack action ===
        attacker_actions_list = []
        for att_idx, att in enumerate(attacker_agents):
            # Attacker generates state from environment (includes fake vehicles from previous steps)
            state = att.get_state()

            # Get attack action from policy network
            action_info = att.get_action(state, test=False)  # Returns ((approach, scale), info)
            attacker_actions_list.append(action_info)

        # === Step 2: Attacker injects fake vehicles into simulation ===
        for idx, (att, action) in enumerate(zip(attacker_agents, attacker_actions_list)):
            if len(action) == 2:
                approach_idx, scale_counts = action
            else:
                continue

            # Convert to approach name and vehicle counts
            approach_name = ['N', 'E', 'S', 'W'][approach_idx % len(att._approaches)]
            vehicle_counts = scale_counts if isinstance(scale_counts, list) else scale_counts.tolist()

            # Inject fake vehicles via world method - this modifies simulation state
            vehicles_injected = self.world.inject_fake_vehicles(
                att.intersection_id,
                approach_name,
                vehicle_counts
            )

            # Update environment state after injection so victim sees fake vehicles
            if vehicles_injected > 0:
                self.world.update_current_measurements()

        # === Step 3: Get victim's poisoned observations after injection ===
        # Victim agent queries environment and gets observation that includes fake vehicles
        if not len(self.agents) == 1:
            victim_observations = [agent.get_ob() for agent in self.agents]
        else:
            victim_observations = [self.agents[0].get_ob()]

        # === Step 4: Victim agent predicts phase based on poisoned observation ===
        if victim_actions is None:
            # Use default action (no phase change) for simplicity
            victim_actions = np.zeros(self.n_agents, dtype=np.int64)

        # === Step 5: Advance simulation with victim's phase decision ===
        self.world.step(victim_actions)

        # === Step 6: Get updated observations and compute attacker reward ===
        if not len(self.agents) == 1:
            # After stepping, get new observations (these are the "next" states for PPO)
            next_observations = [agent.get_ob() for agent in self.agents]
            rewards_attacker = [agent.get_reward() for agent in self.agents]
        else:
            next_observations = [self.agents[0].get_ob()]
            rewards_attacker = [self.agents[0].get_reward()]

        infos = {
            'fake_vehicles_injected': True,
            'attacker_actions': attacker_actions_list,
            'victim_actions_taken': victim_actions,
        }

        return next_observations, victim_observations, rewards_attacker, [False]*self.n_agents, infos

    def reset(self):
        self.world.reset()
        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]  # [agent, sub_agent==1, feature]
            # obs = np.expand_dims(np.array(obs),axis=1)
        else:
            obs = [self.agents[0].get_ob()]  # [agent==1, sub_agent, feature]
        return obs
