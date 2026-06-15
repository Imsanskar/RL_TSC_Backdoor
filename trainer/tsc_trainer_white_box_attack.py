"""
tsc_trainer_white_box_attack.py

Drop-in trainer/evaluator wrapper for a white-box FGSM fake-vehicle attack.

Place this file in trainer/tsc_trainer_white_box_attack.py and place FGSM.py in
attacker/FGSM.py (recommended) or in the project root.  Register the task/trainer
as one of:
    - tsc_white_box_attack
    - tsc_fgsm_fake_vehicle
    - tsc_fgsm

Key difference from a pure feature-space FGSM baseline:
    This wrapper never feeds x_adv directly to the victim policy.  It calls
    FGSM.attack(..., world=self.world), which converts positive FGSM signs into
    fake vehicles on concrete incoming lanes; then the victim re-runs ag.get_ob()
    and chooses its action from the physically poisoned observation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.tsc_trainer import TSCTrainer

from attacker.FGSM import FGSM


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _none_if_string_none(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return value


@Registry.register_trainer("tsc_whitebox")
class TSCTrainerWhiteBox(TSCTrainer):
    """
    TSC trainer/evaluator with white-box FGSM fake-vehicle injection.

    Config keys may be placed under ``attacker_mapping['setting'].param`` or
    ``trainer_mapping['setting'].param``.  Most useful keys:

        fgsm_epsilon / epsilon: float, default 0.007
        fgsm_max_vehicles_per_lane / max_vehicles_per_lane: int, default 10
        fgsm_max_total_vehicles / max_total_vehicles: int or None, default None
        fgsm_top_k_lanes / top_k_lanes: int or None, default None
        fgsm_loss / loss: "ce" | "action_value" | "margin", default "ce"
        fgsm_targeted / targeted: bool, default False
        fgsm_model_attr / model_attr: str or None, default None
        fgsm_strict / strict: bool, default False
        fgsm_attack_train / attack_train: bool, default False
        fgsm_attack_start_episode / attack_start_episode: optional int
        fgsm_attack_end_episode / attack_end_episode: optional int, exclusive
        fgsm_attack_eval / attack_eval: bool, default True
        fgsm_attack_test / attack_test: bool, default True
        fgsm_store_poisoned_obs: bool, default False

    MPLight/FRAP note:
        FGSM.py prepares the exact FRAP input used by MPLightAgent, namely
        [phase | lane_count] or [onehot(phase) | lane_count], and extracts
        Q-values from the PFRL DiscreteActionValue object.

    The attack is applied at decision time.  Fake vehicles are removed before
    the physical rollout, matching the data-injection threat model: fake vehicles
    corrupt the controller's input state but should not affect the subsequent
    controlled rollout.
    """

    def __init__(
        self,
        logger,
        gpu: int = 0,
        cpu: bool = False,
        name: str = "tsc_whitebox",
        wandb=None,
        comet=None,
    ):
        self.fgsm: Optional[FGSM] = None
        self._fgsm_stats: Dict[str, float] = {}
        self._fgsm_last_infos: List[Dict[str, Any]] = []
        self._fgsm_warning_count = 0
        self._fgsm_warning_limit = 5
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name, wandb=wandb, comet=comet)

        # Existing TSCTrainer initializes these after BaseTrainer.create().
        # Attack configuration is therefore initialized here and lazily used in
        # the loops below.
        self._reset_attack_stats()
        self.fgsm_store_poisoned_obs = _as_bool(self._cfg("fgsm_store_poisoned_obs", "store_poisoned_obs", default=False))

    # ------------------------------------------------------------------
    # Optional overrides: create clean TSC components, then prepare FGSM support.
    # ------------------------------------------------------------------
    def create_world(self):
        super().create_world()
        self._prepare_fgsm_world()

    def create_agents(self):
        super().create_agents()
        if _as_bool(self._cfg("fgsm_load_victim_model", "load_victim_model", default=False)):
            self._load_victim_models_if_requested()

    def create_env(self):
        # Keep the standard TSC environment. No PPO attacker object is needed;
        # FGSM uses the world's fake-vehicle API directly when available.
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def _prepare_fgsm_world(self) -> None:
        # SUMO already provides physical insert/remove methods. The FGSM hook
        # helper is a no-op there and only installs lane-count fallback support
        # for worlds that lack native fake-vehicle handling.
        FGSM.ensure_world_fake_vehicle_hooks(self.world)

    # ------------------------------------------------------------------
    # Configuration and attacker initialization
    # ------------------------------------------------------------------
    @staticmethod
    def _param_maps() -> List[Dict[str, Any]]:
        maps: List[Dict[str, Any]] = []
        for key in ("attacker_mapping", "trainer_mapping", "model_mapping", "command_mapping"):
            try:
                maps.append(Registry.mapping[key]["setting"].param)
            except Exception:
                pass
        return maps

    def _cfg(self, *names: str, default: Any = None) -> Any:
        for params in self._param_maps():
            for name in names:
                if name in params:
                    return _none_if_string_none(params[name])
        return default

    def _ensure_attacker(self) -> FGSM:
        if self.fgsm is not None:
            return self.fgsm

        max_total = self._cfg("fgsm_max_total_vehicles", "max_total_vehicles", default=None)
        top_k = self._cfg("fgsm_top_k_lanes", "top_k_lanes", default=None)
        model_attr = self._cfg("fgsm_model_attr", "model_attr", default=None)

        self.fgsm = FGSM(
            epsilon=float(self._cfg("fgsm_epsilon", "epsilon", "eps", default=0.07)),
            max_vehicles_per_lane=int(self._cfg("fgsm_max_vehicles_per_lane", "max_vehicles_per_lane", default=10)),
            max_total_vehicles=None if max_total is None else int(max_total),
            lane_feature_offset=int(self._cfg("fgsm_lane_feature_offset", "lane_feature_offset", default=0)),
            top_k_lanes=None if top_k is None else int(top_k),
            fallback_to_largest_abs_grad=_as_bool(self._cfg("fgsm_fallback", "fallback_to_largest_abs_grad", default=True)),
            min_vehicles_per_selected_lane=int(self._cfg("fgsm_min_vehicles", "min_vehicles_per_selected_lane", default=1)),
            loss=str(self._cfg("fgsm_loss", "loss", default="ce")),
            targeted=_as_bool(self._cfg("fgsm_targeted", "targeted", default=False)),
            model_attr=None if model_attr is None else str(model_attr),
            device=self.device,
            strict=_as_bool(self._cfg("fgsm_strict", "strict", default=False)),
            logger=self.logger,
        )
        self._prepare_fgsm_world()
        return self.fgsm

    def _attack_enabled(self, mode: str, episode: Optional[int] = None) -> bool:
        if mode == "train":
            enabled = _as_bool(self._cfg("fgsm_attack_train", "attack_train", default=False))
            if not enabled:
                return False
            start_ep = self._cfg("fgsm_attack_start_episode", "attack_start_episode", default=None)
            end_ep = self._cfg("fgsm_attack_end_episode", "attack_end_episode", default=None)
            if episode is not None and start_ep is not None and episode < int(start_ep):
                return False
            if episode is not None and end_ep is not None and episode >= int(end_ep):
                return False
            return True
        if mode in {"eval", "val", "train_test"}:
            return _as_bool(self._cfg("fgsm_attack_eval", "attack_eval", default=True))
        if mode == "test":
            return _as_bool(self._cfg("fgsm_attack_test", "attack_test", default=True))
        return False

    def _load_victim_models_if_requested(self) -> None:
        """Best-effort helper for attack-only evaluation from a trained TSC run."""
        try:
            task = Registry.mapping["command_mapping"]["setting"].param.get("task", "")
            base_path = Registry.mapping["logger_mapping"]["path"].path
            model_path = base_path
            for name in ("tsc_white_box_attack", "tsc_fgsm_fake_vehicle", "tsc_fgsm", "tsc_test_white_box_attack"):
                model_path = model_path.replace(name, "tsc")
            explicit = self._cfg("fgsm_victim_model_path", "victim_model_path", default=None)
            if explicit is not None:
                model_path = str(explicit)
            for ag in self.agents:
                try:
                    ag.load_model(e=-1, model_path=model_path)
                except TypeError:
                    ag.load_model(e=-1)
            self.logger.info("Loaded victim model(s) for white-box FGSM attack task %s", task)
        except Exception as exc:
            self.logger.warning("Could not auto-load victim model(s); continuing with current weights. Reason: %s", exc)

    # ------------------------------------------------------------------
    # Attack action selection
    # ------------------------------------------------------------------
    def _select_actions(
        self,
        obs: Sequence[np.ndarray],
        phases: np.ndarray,
        mode: str,
        test: bool = True,
        episode: Optional[int] = None,
    ) -> Tuple[np.ndarray, Sequence[np.ndarray], List[Dict[str, Any]]]:
        """
        Select actions either cleanly or under FGSM fake-vehicle injection.

        Returns
        -------
        actions:
            Array of victim actions, shape [num_agents, sub_agents].
        policy_obs:
            Observations used for action/probability calls.  Under attack this
            is the re-read observation after fake vehicles were injected.
        infos:
            Per-agent FGSM diagnostics.
        """
        if not self._attack_enabled(mode, episode=episode):
            actions = []
            for idx, ag in enumerate(self.agents):
                actions.append(ag.get_action(obs[idx], phases[idx], test=test))
            return np.stack(actions), obs, []

        attacker = self._ensure_attacker()
        attacker.reset_fake_vehicles(self.world)

        clean_actions: List[Any] = []
        infos: List[Dict[str, Any]] = []

        # First compute all gradients on clean observations and inject all fake
        # vehicles into the world.
        for idx, ag in enumerate(self.agents):
            clean_action = ag.get_action(obs[idx], phases[idx], test=True)
            clean_actions.append(clean_action)
            _plan, info = attacker.attack(
                agent=ag,
                obs=obs[idx],
                phase=phases[idx],
                world=self.world,
                clean_action=clean_action,
                inject=True,
                return_info=True,
            )
            infos.append(info)
            if not info.get("success", False):
                self._warn_attack_failure(idx, info)

        # Then force the victim to observe the poisoned world.  This is the step
        # that makes the attack fake-vehicle based rather than feature-space only.
        poisoned_obs = [ag.get_ob() for ag in self.agents]
        actions: List[Any] = []
        for idx, ag in enumerate(self.agents):
            action = ag.get_action(poisoned_obs[idx], phases[idx], test=True)
            actions.append(action)
            info = infos[idx]
            info["clean_action"] = np.asarray(clean_actions[idx]).tolist()
            info["adv_action"] = np.asarray(action).tolist()
            info["changed"] = not np.array_equal(np.asarray(clean_actions[idx]).reshape(-1), np.asarray(action).reshape(-1))

        self._update_attack_stats(infos)
        self._fgsm_last_infos = infos
        return np.stack(actions), poisoned_obs, infos

    def _warn_attack_failure(self, agent_idx: int, info: Dict[str, Any]) -> None:
        if self._fgsm_warning_count >= self._fgsm_warning_limit:
            return
        self._fgsm_warning_count += 1
        self.logger.warning(
            "FGSM fake-vehicle attack produced no injected vehicles for agent %s. Reason: %s",
            agent_idx,
            info.get("error") or "no positive/fallback lane selected",
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def _reset_attack_stats(self) -> None:
        self._fgsm_stats = {
            "decisions": 0.0,
            "agents": 0.0,
            "gradient_successes": 0.0,
            "action_changes": 0.0,
            "fake_vehicles": 0.0,
        }

    def _update_attack_stats(self, infos: Sequence[Dict[str, Any]]) -> None:
        if not infos:
            return
        self._fgsm_stats["decisions"] += 1.0
        self._fgsm_stats["agents"] += float(len(infos))
        for info in infos:
            self._fgsm_stats["gradient_successes"] += 1.0 if info.get("success", False) else 0.0
            self._fgsm_stats["action_changes"] += 1.0 if info.get("changed", False) else 0.0
            self._fgsm_stats["fake_vehicles"] += float(info.get("fake_vehicle_total", 0) or 0)

    def _attack_summary(self) -> Dict[str, float]:
        denom = max(1.0, self._fgsm_stats.get("agents", 0.0))
        decisions = max(1.0, self._fgsm_stats.get("decisions", 0.0))
        return {
            "fgsm_gradient_success_rate": self._fgsm_stats.get("gradient_successes", 0.0) / denom,
            "fgsm_action_change_rate": self._fgsm_stats.get("action_changes", 0.0) / denom,
            "fgsm_fake_vehicles_per_decision": self._fgsm_stats.get("fake_vehicles", 0.0) / decisions,
        }

    # ------------------------------------------------------------------
    # Training and evaluation loops
    # ------------------------------------------------------------------
    def train(self):
        """Train the victim controller; FGSM training attack is off by default."""
        total_decision_num = 0
        flush = 0
        max_travel_time = -float("inf")

        for e in range(self.episodes):
            self.metric.clear()
            last_obs = self.env.reset()
            self._ensure_attacker().reset_fake_vehicles(self.world)
            self._reset_attack_stats()

            for ag in self.agents:
                ag.reset()

            self._set_replay(f"episode_{e}.txt", enabled=(e % self.save_rate == 0))
            episode_loss = []
            i = 0
            dones = [False] * len(self.agents)

            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])

                    if total_decision_num > self.learning_start:
                        actions, policy_obs, _infos = self._select_actions(last_obs, last_phase, mode="train", test=False, episode=e)
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])
                        policy_obs = last_obs

                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                        ob_for_prob = policy_obs[idx] if self.fgsm_store_poisoned_obs else last_obs[idx]
                        try:
                            actions_prob.append(ag.get_action_prob(ob_for_prob, last_phase[idx]))
                        except Exception:
                            actions_prob.append(None)

                    # Fake vehicles should poison only the controller's input;
                    # remove them before rolling the simulator forward.
                    self._ensure_attacker().reset_fake_vehicles(self.world)

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)
                    self.metric.update(rewards)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        replay_last_obs = policy_obs[idx] if self.fgsm_store_poisoned_obs else last_obs[idx]
                        ag.remember(
                            replay_last_obs,
                            last_phase[idx],
                            actions[idx],
                            actions_prob[idx],
                            rewards[idx],
                            obs[idx],
                            cur_phase[idx],
                            dones[idx],
                            f"{e}_{i // self.action_interval}_{ag.id}",
                        )

                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                    total_decision_num += 1
                    last_obs = obs

                if total_decision_num > self.learning_start and total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    cur_loss_q = np.stack([ag.train() for ag in self.agents])
                    episode_loss.append(cur_loss_q)

                if total_decision_num > self.learning_start and total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]

                if all(dones):
                    break

            mean_loss = float(np.mean(np.array(episode_loss))) if episode_loss else 0.0
            summary = self._attack_summary()
            self.writeLog(
                "TRAIN",
                e,
                self.metric.real_average_travel_time(),
                mean_loss,
                self.metric.rewards(),
                self.metric.queue(),
                self.metric.delay(),
                self.metric.throughput(),
            )
            self.logger.info(
                "step:%s/%s, q_loss:%s, rewards:%s, queue:%s, delay:%s, throughput:%s, "
                "fgsm_action_change_rate:%.4f, fgsm_fake_vehicles_per_decision:%.2f",
                i,
                self.steps,
                mean_loss,
                self.metric.rewards(),
                self.metric.queue(),
                self.metric.delay(),
                int(self.metric.throughput()),
                summary["fgsm_action_change_rate"],
                summary["fgsm_fake_vehicles_per_decision"],
            )

            real_travel_time = self.train_test(e)
            if real_travel_time > max_travel_time:
                max_travel_time = real_travel_time
                [ag.save_model(e=e) for ag in self.agents]

            metrics = {
                "Train/Travel Time": self.metric.real_average_travel_time(),
                "Train/Mean Loss": mean_loss,
                "Train/Mean Reward": self.metric.rewards(),
                "Train/Mean Queue": self.metric.queue(),
                "Train/Mean Delay": self.metric.delay(),
                "Train/Throughput": self.metric.throughput(),
                "Attack/FGSM Action Change Rate": summary["fgsm_action_change_rate"],
                "Attack/FGSM Fake Vehicles Per Decision": summary["fgsm_fake_vehicles_per_decision"],
            }
            if self.wandb is not None:
                self.wandb.log({**metrics, "Val/Travel Time": real_travel_time}, step=e)
            if self.comet is not None:
                self.comet.log_metrics({**metrics, "Val/Travel Time": real_travel_time}, step=e)

    def train_test(self, e):
        """Validation after each training episode; attacked by default."""
        obs = self.env.reset()
        self._ensure_attacker().reset_fake_vehicles(self.world)
        self.metric.clear()
        self._reset_attack_stats()

        for ag in self.agents:
            ag.reset()

        dones = [False] * len(self.agents)
        i = 0
        while i < self.test_steps:
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions, _policy_obs, _infos = self._select_actions(obs, phases, mode="eval", test=True)
                self._ensure_attacker().reset_fake_vehicles(self.world)

                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)
                self.metric.update(rewards)
            if all(dones):
                break

        summary = self._attack_summary()
        self.logger.info(
            "Test step:%s/%s, travel time:%s, rewards:%s, queue:%s, delay:%s, throughput:%s, "
            "fgsm_action_change_rate:%.4f, fgsm_fake_vehicles_per_decision:%.2f",
            e,
            self.episodes,
            self.metric.real_average_travel_time(),
            self.metric.rewards(),
            self.metric.queue(),
            self.metric.delay(),
            int(self.metric.throughput()),
            summary["fgsm_action_change_rate"],
            summary["fgsm_fake_vehicles_per_decision"],
        )
        self.writeLog(
            "TEST",
            e,
            self.metric.real_average_travel_time(),
            100,
            self.metric.rewards(),
            self.metric.queue(),
            self.metric.delay(),
            self.metric.throughput(),
        )
        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        """Final test/evaluation; attacked by default."""
        self._set_replay("final.txt", enabled=True)
        self.metric.clear()
        if not drop_load:
            [ag.load_model(self.episodes) for ag in self.agents]

        obs = self.env.reset()
        self._ensure_attacker().reset_fake_vehicles(self.world)
        self._reset_attack_stats()

        for ag in self.agents:
            ag.reset()

        dones = [False] * len(self.agents)
        i = 0
        while i < self.test_steps:
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions, _policy_obs, _infos = self._select_actions(obs, phases, mode="test", test=True)
                self._ensure_attacker().reset_fake_vehicles(self.world)

                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)
                self.metric.update(rewards)
            if all(dones):
                break

        summary = self._attack_summary()
        self.logger.info(
            "Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d, "
            "FGSM action change rate: %.4f, fake vehicles/decision: %.2f",
            self.metric.real_average_travel_time(),
            self.metric.rewards(),
            self.metric.queue(),
            self.metric.delay(),
            self.metric.throughput(),
            summary["fgsm_action_change_rate"],
            summary["fgsm_fake_vehicles_per_decision"],
        )

        metrics = {
            "Test/Travel Time": self.metric.real_average_travel_time(),
            "Test/Mean Reward": self.metric.rewards(),
            "Test/Mean Queue": self.metric.queue(),
            "Test/Mean Delay": self.metric.delay(),
            "Test/Throughput": self.metric.throughput(),
            "Attack/FGSM Action Change Rate": summary["fgsm_action_change_rate"],
            "Attack/FGSM Fake Vehicles Per Decision": summary["fgsm_fake_vehicles_per_decision"],
        }
        if self.wandb is not None:
            self.wandb.log(metrics)
        elif self.comet is not None:
            self.comet.log_metrics(metrics)
        return self.metric

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _set_replay(self, filename: str, enabled: bool = True) -> None:
        try:
            if Registry.mapping["command_mapping"]["setting"].param["world"] == "cityflow":
                if self.save_replay and enabled:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, filename))
                else:
                    self.env.eng.set_save_replay(False)
        except Exception:
            pass
