"""
FGSM.py

White-box Fast Gradient Sign Method (FGSM) for DRL-based traffic-signal-control
agents, physicalized as fake-vehicle injection.

The Haydari et al. attack computes sign(grad_x J(theta, x, a)) with white-box
access to the victim policy/Q-network.  A direct implementation would feed
x_adv = x + epsilon * sign(grad) to the agent.  This implementation instead
uses the FGSM gradient only to decide where fake vehicles should be added, then
updates the simulator world's fake-vehicle counters and forces the victim agent
to re-read its observation from the world.  In other words, the victim action is
chosen from a poisoned traffic observation, not from an arbitrary feature-space
vector.

Expected placement:
    attacker/FGSM.py       (recommended)
    or project root FGSM.py
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from common.registry import Registry


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], torch.Tensor]


@dataclass
class FakeVehiclePlan:
    """A physical attack plan: add fake vehicles to concrete lanes."""

    lane_counts: Dict[str, int] = field(default_factory=dict)
    by_intersection: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return int(sum(max(0, int(v)) for v in self.lane_counts.values()))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lane_counts": dict(self.lane_counts),
            "by_intersection": {k: dict(v) for k, v in self.by_intersection.items()},
            "total": self.total,
        }


@dataclass
class FGSMInfo:
    """Diagnostics returned by FGSM.attack(..., return_info=True)."""

    success: bool
    epsilon: float
    loss: Optional[float] = None
    linf_feature_budget: Optional[float] = None
    model_attr: Optional[str] = None
    forward_signature: Optional[str] = None
    clean_action: Optional[Any] = None
    target_action: Optional[Any] = None
    fake_vehicle_total: int = 0
    fake_vehicle_plan: Dict[str, Any] = field(default_factory=dict)
    gradient_positive_features: int = 0
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "epsilon": self.epsilon,
            "loss": self.loss,
            "linf_feature_budget": self.linf_feature_budget,
            "model_attr": self.model_attr,
            "forward_signature": self.forward_signature,
            "clean_action": self.clean_action,
            "target_action": self.target_action,
            "fake_vehicle_total": self.fake_vehicle_total,
            "fake_vehicle_plan": self.fake_vehicle_plan,
            "gradient_positive_features": self.gradient_positive_features,
            "error": self.error,
        }


def _safe_register_attacker(name: str):
    try:
        return Registry.register_attacker(name)
    except Exception:
        def _wrap(cls):
            return cls
        return _wrap


@_safe_register_attacker("fgsm")
@_safe_register_attacker("FGSM")
@_safe_register_attacker("white_box_fgsm")
class FGSM:
    """
    Physicalized white-box FGSM attacker for TSC agents.

    epsilon is the FGSM magnitude in the observation feature space.  Since this
    code is constrained to add fake vehicles, a positive FGSM sign on a lane
    feature becomes ceil(epsilon * vehicle_max) fake vehicles when vehicle_max is
    available, otherwise at least one fake vehicle.
    """

    DEFAULT_MODEL_ATTRS: Tuple[str, ...] = (
        "model",              # MPLightAgent stores FRAP here.
        "agents_iner.model",  # PFRL DQN wrapper used by MPLight.
        "network", "net", "q_net", "qnet", "q_network",
        "online_net", "eval_net", "policy_net", "policy",
        "actor", "critic", "dqn", "drqn", "estimator", "q_estimator",
    )

    def __init__(
        self,
        epsilon: float = 0.007,
        max_vehicles_per_lane: int = 10,
        max_total_vehicles: Optional[int] = None,
        lane_feature_offset: int = 0,
        top_k_lanes: Optional[int] = None,
        fallback_to_largest_abs_grad: bool = True,
        min_vehicles_per_selected_lane: int = 1,
        loss: str = "ce",
        targeted: bool = False,
        model_attr: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = False,
        logger: Optional[Any] = None,
    ) -> None:
        self.epsilon = float(epsilon)
        self.max_vehicles_per_lane = int(max_vehicles_per_lane)
        self.max_total_vehicles = None if max_total_vehicles is None else int(max_total_vehicles)
        self.lane_feature_offset = int(lane_feature_offset)
        self.top_k_lanes = None if top_k_lanes is None else int(top_k_lanes)
        self.fallback_to_largest_abs_grad = bool(fallback_to_largest_abs_grad)
        self.min_vehicles_per_selected_lane = int(min_vehicles_per_selected_lane)
        self.loss = str(loss)
        self.targeted = bool(targeted)
        self.model_attr = model_attr
        self.device = torch.device(device) if device is not None else None
        self.strict = bool(strict)
        self.logger = logger
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        agent: Any,
        obs: ArrayLike,
        phase: Optional[ArrayLike] = None,
        world: Optional[Any] = None,
        clean_action: Optional[ArrayLike] = None,
        target_action: Optional[ArrayLike] = None,
        inject: bool = True,
        return_info: bool = False,
    ):
        return self.attack(agent, obs, phase, world, clean_action, target_action, inject, return_info)

    def attack(
        self,
        agent: Any,
        obs: ArrayLike,
        phase: Optional[ArrayLike] = None,
        world: Optional[Any] = None,
        clean_action: Optional[ArrayLike] = None,
        target_action: Optional[ArrayLike] = None,
        inject: bool = True,
        return_info: bool = False,
    ):
        """Compute FGSM gradients, convert them to fake vehicles, and inject."""
        if self.epsilon == 0:
            empty = FakeVehiclePlan()
            info = FGSMInfo(success=True, epsilon=0.0, linf_feature_budget=0.0).as_dict()
            return (empty, info) if return_info else empty

        try:
            grad_np, loss_value, clean_action_np, target_action_np, model_attr, signature = self.compute_input_gradient(
                agent=agent,
                obs=obs,
                phase=phase,
                clean_action=clean_action,
                target_action=target_action,
            )

            plan, positive_count = self.gradient_to_fake_vehicle_plan(agent, obs, grad_np)
            injected_total = plan.total
            if inject and world is not None:
                injected_total = self.inject_fake_vehicles(world, plan)

            info_obj = FGSMInfo(
                success=(injected_total > 0),
                epsilon=self.epsilon,
                loss=float(loss_value),
                linf_feature_budget=self.epsilon,
                model_attr=model_attr,
                forward_signature=signature,
                clean_action=self._jsonable(clean_action_np),
                target_action=self._jsonable(target_action_np),
                fake_vehicle_total=int(injected_total),
                fake_vehicle_plan=plan.as_dict(),
                gradient_positive_features=int(positive_count),
            )
            self.last_error = None
            return (plan, info_obj.as_dict()) if return_info else plan

        except Exception as exc:
            self.last_error = str(exc)
            if self.strict:
                raise
            if self.logger is not None:
                try:
                    self.logger.warning("FGSM fake-vehicle attack skipped: %s", self.last_error)
                except Exception:
                    pass
            empty = FakeVehiclePlan()
            info = FGSMInfo(success=False, epsilon=self.epsilon, error=self.last_error).as_dict()
            return (empty, info) if return_info else empty

    def compute_input_gradient(
        self,
        agent: Any,
        obs: ArrayLike,
        phase: Optional[ArrayLike] = None,
        clean_action: Optional[ArrayLike] = None,
        target_action: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray, Optional[np.ndarray], str, str]:
        """Return grad_x J(theta, x, a) with respect to lane-observation x."""
        model, model_attr = self._find_model(agent)
        device = self._infer_device(model)
        self._move_model_to_device(model, device)
        self._sync_wrapped_agent_device(agent, model, device)

        obs_np = self._to_numpy(obs).astype(np.float32, copy=False)
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)

        # IMPORTANT: requires_grad is on raw lane observations only.  For FRAP,
        # phase is concatenated later as a constant, so the returned gradient has
        # the same shape as obs, not [phase | obs].
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device).detach()
        obs_tensor.requires_grad_(True)

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "zero_grad"):
            try:
                model.zero_grad(set_to_none=True)
            except TypeError:
                model.zero_grad()

        with torch.enable_grad():
            scores, signature = self._forward_scores(agent, model, obs_tensor, phase, device)
            scores = self._ensure_2d_scores(scores)
            valid_sets = self._valid_action_sets(agent, obs_np)
            masked_scores = self._mask_invalid_actions(scores, valid_sets)
            clean_target = self._build_action_tensor(masked_scores, clean_action, device, default="argmax", valid_action_sets=valid_sets)
            target_tensor = None
            if self.targeted:
                target_tensor = self._build_action_tensor(masked_scores, target_action, device, default="argmin", valid_action_sets=valid_sets)
            loss = self._build_loss(masked_scores, clean_target, target_tensor)
            loss.backward()

        grad = obs_tensor.grad
        if grad is None:
            raise RuntimeError("Autograd did not produce gradients with respect to the lane observation.")

        if was_training and hasattr(model, "train"):
            model.train()

        clean_np = clean_target.detach().cpu().numpy()
        target_np = None if target_tensor is None else target_tensor.detach().cpu().numpy()
        return grad.detach().cpu().numpy(), float(loss.detach().cpu().item()), clean_np, target_np, model_attr, signature

    def gradient_to_fake_vehicle_plan(
        self,
        agent: Any,
        obs: ArrayLike,
        grad: ArrayLike,
    ) -> Tuple[FakeVehiclePlan, int]:
        """
        Physicalize FGSM signs as fake vehicles on incoming lanes.

        Only positive FGSM signs can be represented directly because this attack
        may add fake vehicles but may not delete real vehicles.
        """
        grad_np = self._to_numpy(grad).astype(np.float32, copy=False)
        if grad_np.ndim == 1:
            grad_np = grad_np.reshape(1, -1)

        generators = self._observation_generators(agent)
        vehicle_max = float(getattr(agent, "vehicle_max", 10.0) or 10.0)
        base_count = int(math.ceil(abs(self.epsilon) * vehicle_max))
        base_count = max(self.min_vehicles_per_selected_lane, base_count)
        base_count = min(base_count, self.max_vehicles_per_lane)

        all_candidates: List[Tuple[float, str, str, int]] = []
        positive_count = 0

        rows = min(len(generators), grad_np.shape[0])
        for row_idx in range(rows):
            inter_id, lanes = generators[row_idx]
            if not lanes:
                continue
            start = self.lane_feature_offset
            stop = min(start + len(lanes), grad_np.shape[1])
            if stop <= start:
                continue
            lane_grad = grad_np[row_idx, start:stop]
            lane_names = lanes[: stop - start]

            pos_idx = np.where(lane_grad > 0)[0]
            positive_count += int(len(pos_idx))
            if len(pos_idx) > 0:
                for local_idx in pos_idx:
                    score = float(abs(lane_grad[int(local_idx)]))
                    all_candidates.append((score, str(inter_id), str(lane_names[int(local_idx)]), base_count))
            elif self.fallback_to_largest_abs_grad and len(lane_grad) > 0:
                local_idx = int(np.argmax(np.abs(lane_grad)))
                score = float(abs(lane_grad[local_idx]))
                all_candidates.append((score, str(inter_id), str(lane_names[local_idx]), base_count))

        all_candidates.sort(key=lambda x: x[0], reverse=True)
        if self.top_k_lanes is not None:
            all_candidates = all_candidates[: max(0, self.top_k_lanes)]

        lane_counts: Dict[str, int] = {}
        by_intersection: Dict[str, Dict[str, int]] = {}
        total = 0
        for _score, inter_id, lane, count in all_candidates:
            if count <= 0:
                continue
            remaining = None if self.max_total_vehicles is None else self.max_total_vehicles - total
            if remaining is not None and remaining <= 0:
                break
            add_count = count if remaining is None else min(count, remaining)
            lane_counts[lane] = lane_counts.get(lane, 0) + int(add_count)
            by_intersection.setdefault(inter_id, {})[lane] = by_intersection.setdefault(inter_id, {}).get(lane, 0) + int(add_count)
            total += int(add_count)

        return FakeVehiclePlan(lane_counts=lane_counts, by_intersection=by_intersection), positive_count

    # ------------------------------------------------------------------
    # Fake-vehicle world hooks
    # ------------------------------------------------------------------
    def inject_fake_vehicles(self, world: Any, plan: FakeVehiclePlan) -> int:
        """Inject a plan into the simulator's fake-vehicle mechanism."""
        self.ensure_world_fake_vehicle_hooks(world)
        if plan.total <= 0:
            return 0

        if self._uses_sumo_vehicle_api(world):
            injected = 0
            for (inter_id, approach), counts in self._sumo_injection_groups(world, plan).items():
                injected += int(world.inject_fake_vehicles(inter_id, approach, counts))
            return injected

        for inter in getattr(world, "intersections", []):
            if not hasattr(inter, "_fgsm_fake_vehicle_lanes"):
                inter._fgsm_fake_vehicle_lanes = {}

        for inter_id, lane_dict in plan.by_intersection.items():
            inter = self._find_intersection(world, inter_id)
            if inter is None:
                continue
            if not hasattr(inter, "_fgsm_fake_vehicle_lanes"):
                inter._fgsm_fake_vehicle_lanes = {}
            for lane, count in lane_dict.items():
                inter._fgsm_fake_vehicle_lanes[lane] = int(inter._fgsm_fake_vehicle_lanes.get(lane, 0)) + int(count)

        if hasattr(world, "_update_infos"):
            world._update_infos()
        return plan.total

    @staticmethod
    def reset_fake_vehicles(world: Any) -> None:
        FGSM.ensure_world_fake_vehicle_hooks(world)
        if hasattr(world, "reset_fake_vehicles"):
            world.reset_fake_vehicles()

    @staticmethod
    def ensure_world_fake_vehicle_hooks(world: Any) -> None:
        """Install/repair lane-level fake vehicle hooks on the world instance."""
        if world is None or getattr(world, "_fgsm_fake_vehicle_hooks_installed", False):
            return
        if FGSM._uses_sumo_vehicle_api(world):
            return

        original_get_fake_affected_lanes = getattr(world, "get_fake_affected_lanes", None)

        def _fgsm_get_fake_affected_lanes(self):
            affected: Dict[str, int] = {}
            if original_get_fake_affected_lanes is not None:
                try:
                    base = original_get_fake_affected_lanes()
                    if isinstance(base, dict):
                        for lane, count in base.items():
                            affected[str(lane)] = affected.get(str(lane), 0) + int(count)
                except Exception:
                    pass

            for inter in getattr(self, "intersections", []):
                direct = getattr(inter, "_fgsm_fake_vehicle_lanes", {})
                if isinstance(direct, dict):
                    for lane, count in direct.items():
                        affected[str(lane)] = affected.get(str(lane), 0) + int(count)
            return affected

        def _fgsm_filter_fake_vehicles(self, raw_data):
            affected = self.get_fake_affected_lanes()
            if not affected or not isinstance(raw_data, dict):
                return raw_data
            filtered: Dict[Any, Any] = {}
            for lane, real_value in raw_data.items():
                fake_count = int(affected.get(str(lane), 0))
                if isinstance(real_value, list):
                    filtered[lane] = list(real_value) + [f"fgsm_fake_{lane}_{i}" for i in range(fake_count)]
                elif isinstance(real_value, (int, float, np.integer, np.floating)):
                    real_float = float(real_value)
                    if math.isnan(real_float):
                        real_float = 0.0
                    filtered[lane] = int(real_float) + fake_count
                else:
                    filtered[lane] = real_value
            return filtered

        def _fgsm_reset_fake_vehicles(self):
            for inter in getattr(self, "intersections", []):
                if hasattr(inter, "fake_vehicles"):
                    inter.fake_vehicles = {}
                inter._fgsm_fake_vehicle_lanes = {}
            if hasattr(self, "_update_infos"):
                self._update_infos()

        world.get_fake_affected_lanes = types.MethodType(_fgsm_get_fake_affected_lanes, world)
        world.filter_fake_vehicles = types.MethodType(_fgsm_filter_fake_vehicles, world)
        world.reset_fake_vehicles = types.MethodType(_fgsm_reset_fake_vehicles, world)
        world._fgsm_fake_vehicle_hooks_installed = True

    @staticmethod
    def _uses_sumo_vehicle_api(world: Any) -> bool:
        return (
            world is not None
            and callable(getattr(world, "inject_fake_vehicles", None))
            and callable(getattr(world, "reset_fake_vehicles", None))
            and callable(getattr(world, "_get_target_road", None))
            and hasattr(world, "net_obj")
        )

    @staticmethod
    def _sumo_injection_groups(world: Any, plan: FakeVehiclePlan) -> Dict[Tuple[str, str], List[int]]:
        groups: Dict[Tuple[str, str], List[int]] = {}
        for inter_id, lane_dict in plan.by_intersection.items():
            inter = FGSM._find_intersection(world, inter_id)
            if inter is None:
                continue

            for lane, count in lane_dict.items():
                if count <= 0:
                    continue
                located = FGSM._sumo_lane_location(world, inter, str(inter_id), str(lane))
                if located is None:
                    continue
                approach, segment_idx = located
                counts = groups.setdefault((str(inter_id), approach), [0, 0, 0])
                counts[segment_idx] += int(count)
        return groups

    @staticmethod
    def _sumo_lane_location(world: Any, inter: Any, inter_id: str, lane: str) -> Optional[Tuple[str, int]]:
        road_lane_mapping = getattr(inter, "road_lane_mapping", {})
        for road, lanes in road_lane_mapping.items():
            lane_list = [str(x) for x in lanes]
            if lane not in lane_list:
                continue

            segment_idx = lane_list.index(lane)
            if segment_idx >= 3:
                return None

            for approach in ("N", "E", "S", "W"):
                try:
                    if str(world._get_target_road(inter_id, approach)) == str(road):
                        return approach, segment_idx
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # Losses and model forwarding
    # ------------------------------------------------------------------
    def _build_loss(
        self,
        scores: torch.Tensor,
        clean_target: torch.Tensor,
        target_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss_name = self.loss.lower()
        if loss_name in {"ce", "cross_entropy", "cross-entropy"}:
            if self.targeted:
                assert target_tensor is not None
                return -F.cross_entropy(scores, target_tensor)
            return F.cross_entropy(scores, clean_target)

        if loss_name in {"action_value", "q", "q_value"}:
            if self.targeted:
                assert target_tensor is not None
                return scores.gather(1, target_tensor.view(-1, 1)).mean()
            return -scores.gather(1, clean_target.view(-1, 1)).mean()

        if loss_name == "margin":
            clean_score = scores.gather(1, clean_target.view(-1, 1)).squeeze(1)
            if self.targeted:
                assert target_tensor is not None
                target_score = scores.gather(1, target_tensor.view(-1, 1)).squeeze(1)
                masked = scores.clone()
                masked.scatter_(1, target_tensor.view(-1, 1), -1e9)
                best_other = masked.max(dim=1).values
                return (target_score - best_other).mean()
            masked = scores.clone()
            masked.scatter_(1, clean_target.view(-1, 1), -1e9)
            best_other = masked.max(dim=1).values
            return (best_other - clean_score).mean()

        raise ValueError(f"Unsupported FGSM loss: {self.loss!r}")

    def _forward_scores(
        self,
        agent: Any,
        model: torch.nn.Module,
        obs_tensor: torch.Tensor,
        phase: Optional[ArrayLike],
        device: torch.device,
    ) -> Tuple[torch.Tensor, str]:
        """Forward the victim and return raw Q/logit scores."""
        attempts: List[Tuple[str, Any]] = []

        # MPLight/FRAP path: get_action() concatenates phase and obs before
        # calling FRAP.forward(states).  FRAP.forward takes exactly one tensor.
        prepared = self._prepare_agent_model_input(agent, obs_tensor, phase, device)
        if prepared is not None:
            model_input, signature = prepared
            attempts.append((signature, lambda model_input=model_input: model(model_input)))

        # Generic graph models.
        edge_idx = getattr(agent, "edge_idx", None)
        if edge_idx is not None:
            edge_idx = edge_idx.to(device) if hasattr(edge_idx, "to") else torch.as_tensor(edge_idx, dtype=torch.long, device=device)
            attempts.extend([
                ("model(x=obs, edge_index=edge_idx, train=True)", lambda: model(x=obs_tensor, edge_index=edge_idx, train=True)),
                ("model(obs, edge_idx, True)", lambda: model(obs_tensor, edge_idx, True)),
            ])

        # Generic models where phase is an explicit argument.
        if phase is not None:
            phase_tensor = torch.as_tensor(self._to_numpy(phase), dtype=torch.long, device=device)
            attempts.extend([
                ("model(obs, phase)", lambda: model(obs_tensor, phase_tensor)),
                ("model(obs, phase, train=True)", lambda: model(obs_tensor, phase_tensor, train=True)),
            ])

        attempts.append(("model(obs)", lambda: model(obs_tensor)))

        errors: List[str] = []
        for signature, fn in attempts:
            try:
                out = fn()
                scores = self._extract_scores(out, device)
                return scores, signature
            except Exception as exc:
                errors.append(f"{signature}: {exc}")
        raise RuntimeError("Could not forward victim model for FGSM. Attempts: " + " | ".join(errors))

    def _prepare_agent_model_input(
        self,
        agent: Any,
        obs_tensor: torch.Tensor,
        phase: Optional[ArrayLike],
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, str]]:
        """
        Build the tensor expected by agents such as MPLight/FRAP.

        MPLightAgent.get_action() does this before calling the PFRL DQN wrapper:
            if phase is enabled and one_hot=False: states = [phase | obs]
            if phase is enabled and one_hot=True:  states = [onehot(phase) | obs]
            otherwise:                            states = obs
        """
        is_frap_like = hasattr(agent, "phase_pairs") or hasattr(agent, "num_actions") or hasattr(agent, "valid_acts")
        if not is_frap_like and phase is None:
            return None

        use_phase = bool(getattr(agent, "phase", False)) and phase is not None
        if not use_phase:
            return obs_tensor, "model(prepared_obs)"

        phase_np = self._to_numpy(phase).reshape(-1).astype(np.int64, copy=False)
        if phase_np.size == 1 and obs_tensor.shape[0] > 1:
            phase_np = np.repeat(phase_np, obs_tensor.shape[0])
        if phase_np.size != obs_tensor.shape[0]:
            phase_np = np.resize(phase_np, obs_tensor.shape[0])
        phase_tensor = torch.as_tensor(phase_np, dtype=torch.long, device=device)

        if bool(getattr(agent, "one_hot", False)):
            n = int(getattr(agent, "num_actions", 0) or len(getattr(agent, "phase_pairs", [])) or int(phase_tensor.max().item()) + 1)
            n = max(1, n)
            phase_clamped = phase_tensor.clamp(0, n - 1)
            phase_onehot = torch.zeros((obs_tensor.shape[0], n), dtype=obs_tensor.dtype, device=device)
            phase_onehot.scatter_(1, phase_clamped.view(-1, 1), 1.0)
            states = torch.cat([phase_onehot, obs_tensor], dim=1)
            return states, "model([onehot(phase) | obs])"

        states = torch.cat([phase_tensor.float().view(-1, 1), obs_tensor], dim=1)
        return states, "model([phase | obs])"

    @staticmethod
    def _extract_scores(out: Any, device: torch.device) -> torch.Tensor:
        """Convert model output/PFRL ActionValue to a score tensor."""
        if torch.is_tensor(out):
            return out
        # PFRL DiscreteActionValueHead returns an ActionValue whose q_values are
        # usually the first element of params.  MPLight batch_act uses params[0].
        if hasattr(out, "params"):
            params = getattr(out, "params")
            if isinstance(params, (tuple, list)) and len(params) > 0:
                q = params[0]
                if torch.is_tensor(q):
                    return q
        if hasattr(out, "q_values") and torch.is_tensor(getattr(out, "q_values")):
            return getattr(out, "q_values")
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return FGSM._extract_scores(out[0], device)
        return torch.as_tensor(out, dtype=torch.float32, device=device)

    @staticmethod
    def _ensure_2d_scores(scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim == 1:
            return scores.view(1, -1)
        if scores.ndim > 2:
            return scores.reshape(-1, scores.shape[-1])
        return scores

    def _valid_action_sets(self, agent: Any, obs_np: np.ndarray) -> Optional[List[Optional[Dict[int, int]]]]:
        """Return per-row valid action maps {internal_q_index: external_action}."""
        valid_acts = getattr(agent, "valid_acts", None)
        if valid_acts is None:
            return None
        rows = int(obs_np.shape[0]) if obs_np.ndim >= 2 else 1
        out: List[Optional[Dict[int, int]]] = []
        keys: List[str] = []
        ob_order = getattr(agent, "ob_order", None)
        if isinstance(ob_order, dict):
            keys = list(ob_order.keys())
        world = getattr(agent, "world", None)
        if not keys and world is not None:
            keys = [str(getattr(x, "id", i)) for i, x in enumerate(getattr(world, "intersections", []))]

        for i in range(rows):
            mapping = None
            if keys and i < len(keys):
                key = keys[i]
                mapping = valid_acts.get(key)
                if mapping is None and not str(key).startswith("GS_"):
                    mapping = valid_acts.get("GS_" + str(key))
                if mapping is None and str(key).startswith("GS_"):
                    mapping = valid_acts.get(str(key)[3:])
            if mapping is None and len(valid_acts) == rows:
                try:
                    mapping = list(valid_acts.values())[i]
                except Exception:
                    mapping = None
            if isinstance(mapping, dict):
                fixed: Dict[int, int] = {}
                for k, v in mapping.items():
                    try:
                        fixed[int(k)] = int(v)
                    except Exception:
                        pass
                out.append(fixed if fixed else None)
            else:
                out.append(None)
        return out

    @staticmethod
    def _mask_invalid_actions(scores: torch.Tensor, valid_action_sets: Optional[List[Optional[Dict[int, int]]]]) -> torch.Tensor:
        if not valid_action_sets:
            return scores
        masked = scores.clone()
        for row, mapping in enumerate(valid_action_sets[: scores.shape[0]]):
            if not mapping:
                continue
            valid_idx = [idx for idx in mapping.keys() if 0 <= int(idx) < scores.shape[1]]
            if not valid_idx:
                continue
            row_mask = torch.full_like(masked[row], -1e9)
            row_mask[torch.as_tensor(valid_idx, dtype=torch.long, device=scores.device)] = 0.0
            masked[row] = masked[row] + row_mask
        return masked

    @staticmethod
    def _build_action_tensor(
        scores: torch.Tensor,
        action: Optional[ArrayLike],
        device: torch.device,
        default: str,
        valid_action_sets: Optional[List[Optional[Dict[int, int]]]] = None,
    ) -> torch.Tensor:
        default_tensor = scores.argmin(dim=1).detach().long() if default == "argmin" else scores.argmax(dim=1).detach().long()
        if action is None:
            return default_tensor

        action_np = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_np.size == 1 and scores.shape[0] > 1:
            action_np = np.repeat(action_np, scores.shape[0])
        if action_np.size != scores.shape[0]:
            return default_tensor

        mapped: List[int] = []
        for row, raw_action in enumerate(action_np):
            chosen = int(raw_action)
            mapping = valid_action_sets[row] if valid_action_sets and row < len(valid_action_sets) else None
            if mapping:
                # MPLight get_action() returns the external phase value. Convert
                # it back to the internal FRAP Q-index if possible.
                reverse = {int(v): int(k) for k, v in mapping.items()}
                if chosen in reverse:
                    chosen = reverse[chosen]
                elif chosen not in mapping:
                    chosen = int(default_tensor[row].item())
            if chosen < 0 or chosen >= scores.shape[1]:
                chosen = int(default_tensor[row].item())
            mapped.append(chosen)
        return torch.as_tensor(mapped, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def _find_model(self, agent: Any) -> Tuple[torch.nn.Module, str]:
        if self.model_attr is not None:
            model = self._resolve_attr(agent, self.model_attr)
            if isinstance(model, torch.nn.Module):
                return model, self.model_attr
            raise AttributeError(f"agent.{self.model_attr} is not a torch.nn.Module")

        for attr in self.DEFAULT_MODEL_ATTRS:
            try:
                model = self._resolve_attr(agent, attr)
            except Exception:
                continue
            if isinstance(model, torch.nn.Module):
                return model, attr
        raise AttributeError("Could not find a torch.nn.Module in the victim agent. Pass model_attr='...'.")

    @staticmethod
    def _resolve_attr(obj: Any, dotted: str) -> Any:
        cur = obj
        for part in dotted.split("."):
            cur = getattr(cur, part)
        return cur

    def _infer_device(self, model: torch.nn.Module) -> torch.device:
        if self.device is not None:
            return self.device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _move_model_to_device(model: torch.nn.Module, device: torch.device) -> None:
        try:
            current = next(model.parameters()).device
        except StopIteration:
            current = None
        if current != device and hasattr(model, "to"):
            model.to(device)

    @staticmethod
    def _sync_wrapped_agent_device(agent: Any, model: torch.nn.Module, device: torch.device) -> None:
        """Keep PFRL-style wrappers consistent after moving the victim model."""
        wrapped = getattr(agent, "agents_iner", None)
        if wrapped is None:
            return

        if getattr(wrapped, "model", None) is model:
            wrapped.device = device
            wrapped.gpu = device.index if device.type == "cuda" else None

        target_model = getattr(wrapped, "target_model", None)
        if isinstance(target_model, torch.nn.Module):
            target_model.to(device)

        optimizer = getattr(wrapped, "optimizer", None)
        if optimizer is not None:
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

    @staticmethod
    def _to_numpy(x: ArrayLike) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _jsonable(x: Any) -> Any:
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    @staticmethod
    def _flatten_lanes(lanes: Any) -> List[str]:
        if lanes is None:
            return []
        out: List[str] = []
        for item in lanes:
            if isinstance(item, (list, tuple)):
                out.extend([str(v) for v in item])
            else:
                out.append(str(item))
        return out

    @staticmethod
    def _ordered_lanes_for_generator(agent: Any, gen: Any) -> List[str]:
        lanes = FGSM._flatten_lanes(getattr(gen, "lanes", []))
        inter = getattr(gen, "I", None)
        inter_id = str(getattr(inter, "id", ""))
        ob_order = getattr(agent, "ob_order", None)
        if not isinstance(ob_order, dict) or not inter_id:
            return lanes

        name = inter_id[3:] if inter_id.startswith("GS_") else inter_id
        lane_order = ob_order.get(name) or ob_order.get(inter_id)
        if not isinstance(lane_order, dict):
            return lanes

        ordered: List[str] = []
        for key, _rank in sorted(lane_order.items(), key=lambda item: item[1]):
            # In the uploaded MPLight code, key is usually an index into tmp.
            lane = None
            try:
                idx = int(key)
                if 0 <= idx < len(lanes):
                    lane = lanes[idx]
            except Exception:
                pass
            if lane is None and str(key) in lanes:
                lane = str(key)
            if lane is not None:
                ordered.append(str(lane))
        return ordered if ordered else lanes

    @staticmethod
    def _observation_generators(agent: Any) -> List[Tuple[str, List[str]]]:
        gens = getattr(agent, "ob_generator", None)
        if gens is None:
            return []
        result: List[Tuple[str, List[str]]] = []
        for idx, item in enumerate(gens):
            gen = item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else item
            inter = getattr(gen, "I", None)
            inter_id = getattr(inter, "id", str(idx))
            lanes = FGSM._ordered_lanes_for_generator(agent, gen)
            result.append((str(inter_id), lanes))
        return result

    @staticmethod
    def _find_intersection(world: Any, inter_id: str) -> Optional[Any]:
        if hasattr(world, "id2intersection") and inter_id in world.id2intersection:
            return world.id2intersection[inter_id]
        for inter in getattr(world, "intersections", []):
            if str(getattr(inter, "id", "")) == str(inter_id):
                return inter
        return None
