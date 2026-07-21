"""
max_queue.py
─────────────────────────────────────────────────────────────────────────────
Simple heuristic agent: at every action interval, give green to the phase
whose incoming lanes have the LONGEST total queue (waiting vehicles).

No learning, no neural network, no replay buffer.

Purpose (from advisor):
  "Check whether MPLight is vulnerable because of the mismatch between the
   reward function and intuitive understanding: longer queue should always
   get a green light."

  → If MPLight achieves higher pressure-based reward BUT worse travel time
    than MaxQueue, the reward function itself is problematic.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import gym

from common.registry import Registry
from agent import utils
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator


@Registry.register_model('maxqueue')
class MaxQueueAgent:
    """
    Pure heuristic: always give green to the phase with the longest queue.

    Compatible with the same trainer interface as MPLight — implements:
        get_ob(), get_reward(), get_phase(), get_queue(), get_delay(),
        get_action(), reset(), load_model(), save_model(),
        remember(), train(), update_target_network()
    """

    def __init__(self, world, rank):
        self.world = world
        self.rank  = rank
        self.sub_agents = len(self.world.intersections)

        # ── Registry lookups ────────────────────────────────────────────────
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['setting']
        self.dic_agent_conf       = Registry.mapping['model_mapping']['setting']

        # ── Phase / signal config ────────────────────────────────────────────
        map_name        = self.dic_traffic_env_conf.param['network']
        self.phase_pairs = self.dic_traffic_env_conf.param['signal_config'][map_name]['phase_pairs']
        self.num_phases  = len(self.phase_pairs)
        self.valid_acts  = self.dic_traffic_env_conf.param['signal_config'][map_name].get('valid_acts')
        self.action_space = gym.spaces.Discrete(self.num_phases)

        # ── Generators ──────────────────────────────────────────────────────
        self.ob_generator      = self._make_generators('lane_count',        average=None,  negative=False)
        self.reward_generator  = self._make_generators('lane_waiting_count', average='all', negative=True)
        self.phase_generator   = self._make_phase_generators()
        self.queue             = self._make_generators('lane_waiting_count', average=None,  negative=False)
        self.delay             = self._make_generators('lane_delay',         average='all', negative=False)

        # ── Decision log ────────────────────────────────────────────────────
        self.decision_log = []
        self._episode_count = 0
        self._step_count    = 0

        # ── Print config info ────────────────────────────────────────────────
        try:
            trainer_param = Registry.mapping['trainer_mapping']['setting'].param
            episodes   = trainer_param.get('episodes',   'N/A')
            steps      = trainer_param.get('steps',      'N/A')
            test_steps = trainer_param.get('test_steps', 'N/A')
            print(f"[MaxQueue] Initialized | episodes={episodes}, steps={steps}, test_steps={test_steps}")
        except Exception:
            print("[MaxQueue] Initialized (could not read trainer config)")

    # ── Generator helpers ────────────────────────────────────────────────────

    def _make_generators(self, feature, average, negative):
        generators = []
        for inter in self.world.intersections:
            node_idx = self.world.id2idx[inter.id]
            gen = LaneVehicleGenerator(
                self.world, inter, [feature],
                in_only=True, average=average, negative=negative
            )
            generators.append((node_idx, gen))
        return sorted(generators, key=lambda x: x[0])

    def _make_phase_generators(self):
        generators = []
        for inter in self.world.intersections:
            node_idx = self.world.id2idx[inter.id]
            gen = IntersectionPhaseGenerator(
                self.world, inter, ['phase'],
                targets=['cur_phase'], negative=False
            )
            generators.append((node_idx, gen))
        return sorted(generators, key=lambda x: x[0])

    # ── Standard interface ───────────────────────────────────────────────────

    def reset(self):
        """Rebuild generators at the start of each episode."""
        self._episode_count += 1
        self._step_count = 0
        print(f"[MaxQueue] Episode {self._episode_count} started")
        self.ob_generator     = self._make_generators('lane_count',         average=None,  negative=False)
        self.reward_generator = self._make_generators('lane_waiting_count',  average='all', negative=True)
        self.phase_generator  = self._make_phase_generators()
        self.queue            = self._make_generators('lane_waiting_count',  average=None,  negative=False)
        self.delay            = self._make_generators('lane_delay',          average='all', negative=False)

    def get_ob(self):
        """Return lane counts per intersection: shape [sub_agents, num_lanes]."""
        return [gen.generate() for _, gen in self.ob_generator]

    def get_reward(self):
        """Return mean negative waiting count (same signal as MPLight reward)."""
        rewards = [gen.generate() for _, gen in self.reward_generator]
        return np.squeeze(np.array(rewards, dtype=np.float32))

    def get_phase(self):
        """Return current phase index per intersection: shape [sub_agents,]."""
        phase = [gen.generate() for _, gen in self.phase_generator]
        return np.concatenate(phase).astype(np.int8)

    def get_queue(self):
        """Return total queue length per intersection."""
        queues = [gen.generate() for _, gen in self.queue]
        tmp = np.squeeze(np.array(queues, dtype=np.float32))
        if self.sub_agents == 1:
            return float(np.sum(tmp))
        return [float(np.sum(x)) for x in tmp]

    def get_delay(self):
        """Return mean delay per intersection."""
        delays = [gen.generate() for _, gen in self.delay]
        return np.squeeze(np.array(delays, dtype=np.float32))

    # ── Core heuristic ───────────────────────────────────────────────────────

    def get_action(self, ob, phase, test=False):
        """
        Heuristic decision: give green to the phase with the longest queue.

        Each phase_pair = list of lane indices that get green together.
        We sum the waiting counts on those lanes and pick the phase
        with the maximum total.

        Parameters
        ----------
        ob    : list of np.array, shape [sub_agents][num_lanes]
                lane-level waiting / vehicle counts from ob_generator
        phase : np.array [sub_agents]  (ignored — heuristic is stateless)
        test  : bool (ignored — behaviour is identical train/test)

        Returns
        -------
        acts : np.array [sub_agents]  chosen phase index per intersection
        """
        # Get per-intersection waiting counts
        waiting = [gen.generate() for _, gen in self.queue]

        self._step_count += 1
        if self._step_count % 100 == 0:
            print(f"[MaxQueue] Episode {self._episode_count} | Step {self._step_count}")

        acts = []
        queue_per_phase_all = []   # for logging

        for idx in range(self.sub_agents):
            w = np.array(waiting[idx], dtype=float)   # [num_lanes]
            num_lanes = len(w)

            # Sum waiting vehicles for each phase's lane pairs
            phase_queues = []
            for pair in self.phase_pairs:
                # pair is a list of lane indices belonging to this phase
                valid = [p for p in pair if p < num_lanes]
                phase_queues.append(float(np.sum(w[valid])) if valid else 0.0)

            chosen = int(np.argmax(phase_queues))
            acts.append(chosen)
            queue_per_phase_all.append(phase_queues)

        acts = np.array(acts)
        self._log_decision(ob, phase, acts, queue_per_phase_all)
        return acts

    # ── Decision logging ─────────────────────────────────────────────────────

    def _log_decision(self, ob, phase, acts, queue_per_phase_all):
        """Log one row per intersection per decision step."""
        try:
            sim_time = self.world.eng.simulation.getTime()
        except Exception:
            sim_time = None

        for idx in range(self.sub_agents):
            inter_id = self.world.intersection_ids[idx]
            chosen   = int(acts[idx])

            sig_state = ""
            try:
                sig_state = self.world.eng.trafficlight.getRedYellowGreenState(inter_id)
            except Exception:
                pass

            ob_for_inter = ob[idx] if idx < len(ob) else None
            total_real_vehicles = float(np.sum(ob_for_inter)) if ob_for_inter is not None else None

            row = {
                'time':                   sim_time,
                'intersection':           inter_id,
                'current_phase_input':    int(phase[idx]) if idx < len(phase) else None,
                'chosen_phase':           chosen,
                'max_queue_phase':        chosen,   # same — heuristic always picks max queue
                'divergence_vs_maxqueue': 0,        # by definition always 0
                'real_vehicles_on_inlanes': total_real_vehicles,
                'signal_state':           sig_state,
            }
            # per-phase queue totals
            qpph = queue_per_phase_all[idx] if idx < len(queue_per_phase_all) else []
            for i, q in enumerate(qpph):
                row[f'queue_phase_{i}'] = float(q)

            self.decision_log.append(row)

    def save_decision_log_to_excel(self, path):
        """Write accumulated decision log to .xlsx."""
        if not self.decision_log:
            print(f"[MaxQueue] decision_log is empty, skipping {path}")
            return
        try:
            import pandas as pd
            df = pd.DataFrame(self.decision_log)
            df.to_excel(path, index=False)
            print(f"[MaxQueue] Decision log saved → {path}")
        except Exception as exc:
            print(f"[MaxQueue] Could not save decision log: {exc}")
        import os
        import torch
        """
        Write a dummy checkpoint. MaxQueue has no learned weights, but the
        attacker's load_model expects best_<rank>.pth to exist.
        """
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        os.makedirs(path, exist_ok=True)
        model_name = os.path.join(path, f'best_{self.rank}.pth')  # NOTE: .pth
        # torch.save({'agent': 'maxqueue', 'rank': self.rank}, model_name)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, model_name)
        print(f"[MaxQueue] Dummy model saved → {model_name}")

    # ── No-op stubs (required by trainer interface) ──────────────────────────

    def remember(self, last_obs, last_phase, actions, actions_prob,
                 rewards, obs, cur_phase, done, key):
        """No replay buffer — heuristic does not learn."""
        pass

    def train(self):
        """No training — heuristic does not learn."""
        pass

    def update_target_network(self):
        """No target network — heuristic does not learn."""
        pass

    def load_model(self, e, model_path=None):
        """No model weights to load — MaxQueue is a heuristic."""
        pass

