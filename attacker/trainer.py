"""
Attacker Trainer - Training loop for Multi-PPO adversarial attack.

This module implements the training loop for the adversarial attacker that
manipulates traffic perception to maximize traffic delay.
"""

import os
import sys
import numpy as np
from common.metrics import Metrics
from common.registry import Registry

# Import BaseTrainer from parent trainer module
sys.path.insert(0, '/data/srg/samgain/RL_ITS/g2p-tsc')
from trainer.base_trainer import BaseTrainer


@Registry.register_trainer("tsc_attacker")
class TSCTrainerAttacker(BaseTrainer):
    """
    Trainer for Multi-PPO adversarial attack on traffic signal control.

    The attacker:
    1. Observes traffic state
    2. Selects attack actions (which approach + fake vehicle counts)
    3. Injects fake vehicles into perception
    4. Victim controller makes signal decision
    5. Attacker receives reward based on delay caused
    """

    def __init__(self, logger, gpu=0, cpu=False, name="tsc_attacker", wandb=None, comet=None):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        # Get parameters from config
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param.get('learning_start', 1000)
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param.get('update_model_rate', 100)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param.get('yellow_length', 5)

        # Attacker-specific parameters
        self.max_fake_vehicles = Registry.mapping['model_mapping']['setting'].param.get('max_fake_vehicles', 10)
        self.penalty_lambda = Registry.mapping['model_mapping']['setting'].param.get('penalty_lambda', 0.01)
        self.num_segments = Registry.mapping['model_mapping']['setting'].param.get('num_segments', 3)

        self.wandb = wandb
        self.comet = comet

        # Log file path
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     'attacker_log.log')

        # Metrics for evaluation
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, None, [])

    def create_world(self):
        """Create world with CityFlow engine."""
        self.world = Registry.mapping['world_mapping'][
            Registry.mapping['command_mapping']['setting'].param['world']
        ](self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],
          interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_agents(self):
        """Create the attacker agent."""
        self.agents = []
        agent = Registry.mapping['model_mapping'][
            Registry.mapping['command_mapping']['setting'].param['agent']
        ](self.world, 0, param=Registry.mapping['model_mapping']['setting'].param)
        self.agents.append(agent)

    def train(self):
        """
        Train the attacker.

        For each episode:
        1. Reset world and attacker
        2. For each timestep:
           - Attacker observes state
           - Attacker selects action (approach + fake vehicle counts)
           - Inject fake vehicles
           - Victim controller acts (we use random actions)
           - Get reward (negative delay)
           - Update attacker policy
        """
        total_steps = 0
        min_delay = float('inf')

        for e in range(self.episodes):
            # Reset world
            self.world.reset()
            self.metric.update([])  # Clear metrics

            # Reset attacker
            for a in self.agents:
                a.reset()

            episode_rewards = []
            episode_fake_vehicles = []
            i = 0

            while i < self.steps:
                # Get current world state
                lane_counts = self.world.eng.get_lane_vehicle_count()
                lane_speeds = self.world.eng.get_vehicle_speed()
                current_phases = self.world.get_cur_phase()

                # Get attacker state and action for each intersection
                approach_actions = []
                scale_actions = []
                states = []

                for idx, ag in enumerate(self.agents):
                    state = ag.get_state()
                    states.append(state)
                    action, _ = ag.get_action(state, test=False)
                    approach_actions.append(action[0])
                    scale_actions.append(action[1])

                # Reset fake vehicles at start of each action interval
                if self.world is not None:
                    self.world.reset_fake_vehicles()

                # Inject fake vehicles using SDSM injector from attacker (more reliable)
                total_fake = 0
                for idx, ag in enumerate(self.agents):
                    approach = approach_actions[idx]
                    scales = scale_actions[idx]
                    num_fakes = ag.injector.inject_approach_vehicles(
                        ag._approaches[approach % len(ag._approaches)],
                        scales
                    )
                    total_fake += num_fakes

                # Victim controller takes actions (random for now)
                # In real scenario, this would be the pre-trained victim controller
                actions = np.random.randint(0, 4, len(self.agents))  # Random phase

                # Step the world
                for _ in range(self.action_interval):
                    self.world.step(actions)
                    self.world.update_current_measurements()
                    i += 1

                    # Calculate metrics
                    real_delay = self.world.eng.get_average_travel_time()
                    throughput = self.world.get_cur_throughput()

                    # Calculate attacker reward
                    total_delay = sum(lane_counts.get(l, 0) for l in self.world.all_lanes)
                    attacker_reward = -total_delay - self.penalty_lambda * total_fake

                    episode_rewards.append(attacker_reward)
                    episode_fake_vehicles.append(total_fake)

                    # Update policy periodically
                    if total_steps > self.learning_start and total_steps % self.update_model_rate == 0:
                        for ag in self.agents:
                            losses = ag.update_policy()
                            if losses:
                                print(f"Step {total_steps}: Actor Loss={losses.get('actor_loss', 0):.4f}, "
                                      f"Critic Loss={losses.get('critic_loss', 0):.4f}")

                    total_steps += 1

                    if i >= self.steps:
                        break

                if i >= self.steps:
                    break

            # Log episode statistics
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0
            mean_fake = np.mean(episode_fake_vehicles) if episode_fake_vehicles else 0

            # Get final metrics
            final_delay = self.world.eng.get_average_travel_time()

            self.logger.info(f"Episode {e}: Reward={mean_reward:.4f}, Fake Vehicles={mean_fake:.2f}, "
                           f"Delay={final_delay:.4f}")

            # Save model if improved
            if final_delay < min_delay:
                min_delay = final_delay
                for ag in self.agents:
                    ag.save_model(os.path.join(
                        Registry.mapping['logger_mapping']['path'].path,
                        'model', f'best_attacker_{e}.pt'
                    ))

            # Log to wandb/comet
            metrics = {
                'Train/Reward': mean_reward,
                'Train/FakeVehicles': mean_fake,
                'Train/Delay': final_delay,
                'Train/TravelTime': final_delay
            }

            if self.wandb is not None:
                self.wandb.log({**metrics}, step=e)
            elif self.comet is not None:
                self.comet.log_metrics({**metrics}, step=e)

    def train_test(self, e):
        """Evaluate model after each training episode."""
        # This is a placeholder for compatibility with BaseTrainer
        # For the attacker, we don't need a separate train_test since
        # we evaluate inline during training
        return 0.0

    def test(self, drop_load=True):
        """Test the attacker on pre-trained victim controller."""
        # Reset world
        self.world.reset()
        self.metric.update([])

        if not drop_load:
            for ag in self.agents:
                ag.load_model(self.episodes)

        for a in self.agents:
            a.reset()

        episode_rewards = []
        episode_fake_vehicles = []

        for i in range(self.test_steps):
            # Get current world state
            lane_counts = self.world.eng.get_lane_vehicle_count()
            lane_speeds = self.world.eng.get_vehicle_speed()
            current_phases = self.world.get_cur_phase()

            # Get attacker action
            approach_actions = []
            scale_actions = []

            for idx, ag in enumerate(self.agents):
                state = ag.get_state()
                action, _ = ag.get_action(state, test=True)
                approach_actions.append(action[0])
                scale_actions.append(action[1])

            # Reset and inject fake vehicles
            for ag in self.agents:
                self.world.reset_fake_vehicles(ag.intersection_id)

            total_fake = 0
            for idx, ag in enumerate(self.agents):
                approach = approach_actions[idx]
                scales = scale_actions[idx]
                num_fakes = self.world.inject_fake_vehicles(
                    ag.intersection_id,
                    ag._approaches[approach % len(ag._approaches)],
                    scales
                )
                total_fake += num_fakes

            # Victim controller takes actions
            actions = np.random.randint(0, 4, len(self.agents))

            # Step the world
            for _ in range(self.action_interval):
                self.world.step(actions)
                self.world.update_current_measurements()

                # Calculate reward
                real_delay = self.world.eng.get_average_travel_time()
                total_delay = sum(lane_counts.get(l, 0) for l in self.world.all_lanes)
                attacker_reward = -total_delay - self.penalty_lambda * total_fake

                episode_rewards.append(attacker_reward)
                episode_fake_vehicles.append(total_fake)

                i += 1
                if i >= self.test_steps:
                    break

        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        mean_fake = np.mean(episode_fake_vehicles) if episode_fake_vehicles else 0
        final_delay = self.world.eng.get_average_travel_time()

        self.logger.info(f"Test - Final Travel Time: {final_delay:.4f}, "
                       f"Reward: {mean_reward:.4f}, Fake Vehicles: {mean_fake:.2f}")

        if self.wandb is not None:
            self.wandb.log({
                'Test/Travel Time': final_delay,
                'Test/Delay': final_delay,
                'Test/Reward': mean_reward,
                'Test/FakeVehicles': mean_fake
            })
        elif self.comet is not None:
            self.comet.log_metrics({
                'Test/Travel Time': final_delay,
                'Test/Delay': final_delay,
                'Test/Reward': mean_reward,
                'Test/FakeVehicles': mean_fake
            })

        return self.metric


@Registry.register_trainer("tsc_attacker_test")
class TSCTesterAttacker(TSCTrainerAttacker):
    """Tester for pre-trained attacker."""

    def test(self):
        """Test attacker on victim controller."""
        # Same as parent test method
        return super().test(drop_load=True)
