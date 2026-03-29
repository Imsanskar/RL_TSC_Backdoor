"""
Multi-PPO Attacker - Integrated with TSC Environment.

This module implements a multi-action PPO attacker that:
1. Observes traffic state from victim environment
2. Selects attack actions (approach + vehicle counts)
3. Injects fake vehicles to poison perception
4. Gets updated observation from modified environment
5. Computes reward based on traffic impact
"""

import numpy as np
import torch
import torch.nn as nn
from common.registry import Registry


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layers with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@Registry.register_model('multi_ppo_attacker')
class MultiPPOAttacker:
    """
    Multi-PPO attacker integrated with TSC environment.

    The attacker uses three networks:
    1. Critic: Estimates state-value function V(s)
    2. Approach Actor: Selects which road to target (N/E/S/W)
    3. Scale Actor: Selects fake vehicle counts per segment

    Attack flow:
    1. Observe current traffic state from environment
    2. Get attack action from policy network
    3. Inject fake vehicles via SDSM injector
    4. Step environment to get updated observation
    5. Compute reward from traffic metrics
    """

    def __init__(self, world, rank, state_dim=None, **kwargs):
        """
        Initialize Multi-PPO attacker.

        Args:
            world: World object with CityFlow engine
            rank: Rank of this attacker (for multi-attacker setups)
            state_dim: State dimension (will be computed from state_generator)
            **kwargs: Additional parameters from config
        """
        self.world = world
        self.rank = rank
        self.intersection_id = world.intersection_ids[rank]

        # Get parameters from config
        param = kwargs.get('param', {})
        self.actor_learning_rate = param.get('actor_learning_rate', 1e-4)
        self.critic_learning_rate = param.get('critic_learning_rate', 1e-4)
        self.gamma = param.get('gamma', 0.99)
        self.gae_lambda = param.get('gae_lambda', 0.95)
        self.clip_epsilon = param.get('clip_epsilon', 0.1)
        self.value_coef = param.get('value_coef', 0.5)
        self.entropy_coef = param.get('entropy_coef', 0.01)
        self.max_grad_norm = param.get('max_grad_norm', 0.5)
        self.num_segments = param.get('num_segments', 3)
        self.num_approaches = param.get('num_approaches', 4)
        self.max_vehicles_per_segment = param.get('max_vehicles_per_segment', 10)
        self.penalty_lambda = param.get('penalty_lambda', 0.01)
        device_name = Registry.mapping['command_mapping']['setting'].param['device']
        if device_name != 'cpu' and not torch.cuda.is_available():
            device_name = 'cpu'
        self.device = torch.device(device_name)

        # Import and initialize state generator
        from attacker.state_generator import AttackerStateGenerator

        self.state_gen = AttackerStateGenerator(
            world, world.id2intersection[self.intersection_id],
            num_segments=self.num_segments,
            num_approaches=self.num_approaches
        )

        # SDSM Injector for fake vehicle injection
        from attacker.sdsm_injector import SDSMInjector

        self.injector = SDSMInjector(
            world, self.intersection_id,
            max_vehicles_per_segment=self.max_vehicles_per_segment,
            num_segments=self.num_segments,
            penalty_lambda=self.penalty_lambda
        )

        # State dimension (from state generator)
        self.state_dim = self.state_gen.ob_length

        # Build networks
        self.actor = MultiPPOActor(
            state_dim=self.state_dim,
            num_approaches=self.num_approaches,
            max_vehicles=self.max_vehicles_per_segment,
            num_segments=self.num_segments
        ).to(self.device)
        self.critic = MultiPPOCritic(state_dim=self.state_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        # Training buffers
        self.replay_buffer = []
        self.max_buffer_size = 256

        # Exploration rate (for epsilon-greedy during training)
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        # Injection state tracking
        self.current_plan = None

        # Approaches mapping
        self._approaches = ['N', 'E', 'S', 'W']

    def reset(self):
        """Reset attacker state."""
        self.replay_buffer = []
        self.epsilon = 1.0
        self.current_plan = None
        self.injector.reset()

    def get_state(self):
        """
        Get current state observation from environment.

        Returns:
            State vector as numpy array
        """
        return self.state_gen.generate()

    def get_action(self, state, test=False):
        """
        Get attacker action for given state using policy network.

        Args:
            state: Current state observation (numpy array)
            test: If True, use deterministic action selection

        Returns:
            Tuple of (approach_action, scale_action) and optional action info dict
        """

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Sample approach 
        logits, scale_mean = self.actor(state_tensor)

        approach_dist = torch.distributions.Categorical(logits=logits)

        if test:
            approach_action = torch.argmax(logits, dim=-1)
        else:
            approach_action = approach_dist.sample()

        scale_std = torch.ones_like(scale_mean) * 0.3
        scale_dist = torch.distributions.Normal(scale_mean, scale_std)

        if test:
            scale_action = scale_mean
        else:
            scale_action = scale_dist.sample()

         # ===== LOG PROB =====
        log_prob = (
            approach_dist.log_prob(approach_action)
            + scale_dist.log_prob(scale_action).sum(dim=-1)
        )

        # ===== VALUE =====
        with torch.no_grad():
            value = self.critic(state_tensor).squeeze(-1)

        # ===== FORMAT =====
        approach_action = approach_action.item()

        scale_action = (
            scale_action.squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        scale_action = np.clip(
            scale_action,
            0,
            self.max_vehicles_per_segment
        )

        return (approach_action, scale_action), log_prob.item(), value.item()

    def get_value(self, state):
        """Get state value estimate."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.critic(state_tensor).squeeze(0).cpu().numpy()



    def compute_advantage(self, rewards, values, next_values, dones):
        """
        Compute generalized advantage estimate (GAE).

        Args:
            rewards: List of rewards
            values: List of state values
            next_values: List of next state values
            dones: List of done flags

        Returns:
            List of advantages
        """
        advantages = []
        for i in range(len(rewards)):
            # Compute TD error
            td_error = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            advantages.append(td_error)

        return advantages

    def train(self, batch):
        """
        Train attacker on a batch of transitions.

        Args:
            batch: List of (state, approach_action, scale_action, reward, next_state, done) tuples
`
        Returns:
            Dictionary of training losses
        """
        if len(batch) == 0:
            return {'critic_loss': 0.0, 'actor_loss': 0.0, 'value_loss': 0.0}

        # Unpack batch
        states = torch.as_tensor(np.array([t[0] for t in batch]), dtype=torch.float32, device=self.device)
        approach_actions = torch.as_tensor([t[1][0] for t in batch], dtype=torch.long, device=self.device)
        scale_actions = torch.as_tensor([t[1][1] for t in batch], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([t[2] for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array([t[3] for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t[4] for t in batch], dtype=torch.bool, device=self.device)
        old_log_probs = torch.as_tensor([t[5] for t in batch], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor([t[6] for t in batch], dtype=torch.float32, device=self.device)


        # normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute returns and advantages
        values = self.critic(states)
        next_values = self.critic(next_states).detach()
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1).float()
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train critic (loss over time steps)
        critic_loss = torch.nn.functional.mse_loss(values, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Train actor
        approach_logits, scale_means = self.actor(states)
        approach_dist = torch.distributions.Categorical(logits=approach_logits)
        approach_log_probs = approach_dist.log_prob(approach_actions).unsqueeze(1)

        scale_dist = torch.distributions.Normal(scale_means, torch.ones_like(scale_means) * 0.5)
        scale_log_probs = scale_dist.log_prob(scale_actions).sum(dim=1, keepdim=True)

        total_log_probs = approach_log_probs + scale_log_probs

        # PPO ratio and clipped surrogate loss
        ratios = torch.exp(total_log_probs - old_log_probs.view((total_log_probs.shape[0], -1)).detach())
        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        # Entropy bonus
        entropy = approach_dist.entropy().mean() + scale_dist.entropy().mean()
        actor_loss = actor_loss - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy': entropy.item()
        }

    def update_policy(self, num_updates=1):
        """
        Update attacker policy from replay buffer.

        Args:
            num_updates: Number of training updates to perform

        Returns:
            Average losses across updates
        """
        if len(self.replay_buffer) < self.max_buffer_size:
            return

        # Sample batch
        batch_size = min(self.max_buffer_size, len(self.replay_buffer))
        batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch]

        losses = []
        for _ in range(num_updates):
            batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in batch]
            loss_dict = self.train(batch)
            losses.append(loss_dict)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {k: np.mean([l[k] for l in losses]) for k in losses[0]}

    def observe(self, state, action, log_prob_value, reward, next_state, done):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: (approach_action, scale_action)
            reward: Reward received
            next_state: Next state after attack effect
            done: Episode done
        """
        log_prob, value = log_prob_value
        self.replay_buffer.append((state, action, reward, next_state, done, log_prob, value))

        # Trim buffer if too large
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]

    def save_model(self, path):
        """Save attacker model to path."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """Load attacker model from path."""
        checkpoint = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        self.actor.to(self.device)
        self.critic.to(self.device)

    def cleanup(self):
        """Remove all injected fake vehicles from simulation."""
        self.injector.cleanup_injected_vehicles()



class MultiPPOActor(nn.Module):
    """
    Actor network with separate approach and scale heads.

    Two independent policies:
    - Approach: Discrete action (which road to target)
    - Scale: Continuous action (vehicle counts per segment)
    """

    def __init__(self, state_dim, num_approaches, max_vehicles, num_segments):
        super().__init__()
        self.num_approaches = num_approaches
        self.max_vehicles = max_vehicles
        self.num_segments = num_segments

        # Shared encoder
        self.encoder_approach = nn.Sequential(
            layer_init(nn.Linear(state_dim, state_dim * 3)),
            nn.ReLU(),

            layer_init(nn.Linear(state_dim * 3, 128)),
            nn.ReLU(),

            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),

            layer_init(nn.Linear(256, 256)),
            nn.ReLU()
        )

        self.encoder_actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, state_dim * 3)),
            nn.ReLU(),

            layer_init(nn.Linear(state_dim * 3, 128)),
            nn.ReLU(),

            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),

            layer_init(nn.Linear(256, 256)),
            nn.ReLU()
        )

        # Approach actor (discrete - Categorical policy)
        self.approach_actor = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, num_approaches), std=0.01)
        )

        # Scale actor (continuous - outputs mean per segment)
        self.scale_actor = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, num_segments), std=0.01),
            nn.Sigmoid()  # Output between 0 and max_vehicles
        )

    def forward(self, state):
        x = self.encoder_approach(state)
        approach_logits = self.approach_actor(x)
        scale_mean = self.scale_actor(self.encoder_actor(state)) * self.max_vehicles
        return approach_logits, scale_mean



class MultiPPOCritic(nn.Module):
    """Critic network for state-value estimation."""

    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, state_dim * 3)),
            nn.ReLU(),

            layer_init(nn.Linear(state_dim * 3, 128)),
            nn.ReLU(),

            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),

            layer_init(nn.Linear(256, 512)),
            nn.ReLU(),

            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),

            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),

            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def forward(self, state):
        return self.network(state)

