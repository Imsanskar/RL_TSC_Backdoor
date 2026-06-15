import math
import os
import time

import numpy as np
from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.base_trainer import BaseTrainer


@Registry.register_trainer("tsc_min_count_adversarial")
class TSCTrainerMinCountAdversarial(BaseTrainer):
    """
    SUMO-based deterministic fake-vehicle attacker.

    At every victim decision point, each intersection's incoming lane with the
    fewest vehicles is selected. The full fake-vehicle budget is injected into
    that concrete SUMO lane through world_sumo.inject_fake_vehicles().
    """

    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc",
        wandb=None,
        comet=None,
    ):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.episodes = Registry.mapping["trainer_mapping"]["setting"].param["episodes"]
        self.steps = Registry.mapping["trainer_mapping"]["setting"].param["steps"]
        self.test_steps = Registry.mapping["trainer_mapping"]["setting"].param["test_steps"]
        self.buffer_size = Registry.mapping["trainer_mapping"]["setting"].param["buffer_size"]
        self.action_interval = Registry.mapping["trainer_mapping"]["setting"].param["action_interval"]
        self.save_rate = Registry.mapping["logger_mapping"]["setting"].param["save_rate"]
        self.learning_start = Registry.mapping["trainer_mapping"]["setting"].param["learning_start"]
        self.update_model_rate = Registry.mapping["trainer_mapping"]["setting"].param["update_model_rate"]
        self.update_target_rate = Registry.mapping["trainer_mapping"]["setting"].param["update_target_rate"]
        self.test_when_train = Registry.mapping["trainer_mapping"]["setting"].param["test_when_train"]
        self.yellow_time = Registry.mapping["trainer_mapping"]["setting"].param["yellow_length"]
        self.log_file = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            Registry.mapping["logger_mapping"]["setting"].param["log_dir"],
            os.path.basename(self.logger.handlers[-1].baseFilename).rstrip("_BRF.log") + "_DTL.log",
        )

        attacker_config = Registry.mapping.get("attacker_mapping", {}).get("setting")
        attacker_param = attacker_config.param if attacker_config is not None else {}
        self.num_segments = int(attacker_param.get("num_segments", 3))
        self.max_vehicles_per_segment = int(attacker_param.get("max_vehicles_per_segment", 10))
        self.fake_vehicle_budget = int(
            attacker_param.get(
                "fake_vehicle_budget",
                self.max_vehicles_per_segment,
            )
        )
        self.wandb = wandb
        self.comet = comet

    def create_world(self):
        self.world = Registry.mapping["world_mapping"][Registry.mapping["command_mapping"]["setting"].param["world"]](
            self.path,
            Registry.mapping["command_mapping"]["setting"].param["thread_num"],
            interface=Registry.mapping["command_mapping"]["setting"].param["interface"],
        )

    def create_metrics(self):
        if Registry.mapping["command_mapping"]["setting"].param["delay_type"] == "apx":
            lane_metrics = ["rewards", "queue", "delay"]
            world_metrics = ["real avg travel time", "throughput"]
        else:
            lane_metrics = ["rewards", "queue"]
            world_metrics = ["delay", "real avg travel time", "throughput"]
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        self.agents = []
        agent = Registry.mapping["model_mapping"][Registry.mapping["command_mapping"]["setting"].param["agent"]](
            self.world, 0
        )
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)
        for i in range(1, num_agent):
            self.agents.append(
                Registry.mapping["model_mapping"][Registry.mapping["command_mapping"]["setting"].param["agent"]](
                    self.world, i
                )
            )

        if Registry.mapping["model_mapping"]["setting"].param["name"] == "magd":
            for ag in self.agents:
                ag.link_agents(self.agents)

        self.n_agents = len(self.world.intersection_ids)
        self.network_model_path = self._controller_model_path()
        for ag in self.agents:
            ag.load_model(e=-1, model_path=self.network_model_path)

    def create_env(self):
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        total_decision_num = 0
        flush = 0
        min_travel_time = float("inf")
        for e in range(self.episodes):
            steps_run, episode_loss, total_decision_num, flush = self._run_attacked_episode(
                self.steps,
                training=True,
                episode=e,
                total_decision_num=total_decision_num,
                flush=flush,
            )
            mean_loss = float(np.mean(np.array(episode_loss))) if episode_loss else 0.0

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
                "step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                    steps_run,
                    self.steps,
                    mean_loss,
                    self.metric.rewards(),
                    self.metric.queue(),
                    self.metric.delay(),
                    int(self.metric.throughput()),
                )
            )
            self.logger.info(
                "episode:{}/{}, real avg travel time:{}".format(
                    e, self.episodes, self.metric.real_average_travel_time()
                )
            )
            for j in range(len(self.world.intersections)):
                self.logger.debug(
                    "intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(
                        j, self.metric.lane_rewards()[j], self.metric.lane_queue()[j]
                    )
                )

            metrics = {
                "Train/Travel Time": self.metric.real_average_travel_time(),
                "Train/Mean Loss": mean_loss,
                "Train/Mean Reward": self.metric.rewards(),
                "Train/Mean Queue": self.metric.queue(),
                "Train/Mean Delay": self.metric.delay(),
                "Train/Throughput": self.metric.throughput(),
            }

            real_travel_time = self.train_test(e)
            if real_travel_time < min_travel_time:
                min_travel_time = real_travel_time

            if self.wandb is not None:
                self.wandb.log({**metrics, "Val/Travel Time": real_travel_time}, step=e)
            if self.comet is not None:
                self.comet.log_metrics({**metrics, "Val/Travel Time": real_travel_time}, step=e)

    def train_test(self, e):
        self._run_attacked_episode(self.test_steps)
        self.logger.info(
            "Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                e,
                self.episodes,
                self.metric.real_average_travel_time(),
                self.metric.rewards(),
                self.metric.queue(),
                self.metric.delay(),
                int(self.metric.throughput()),
            )
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
        if Registry.mapping["command_mapping"]["setting"].param["world"] == "cityflow":
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, "final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()
        if not drop_load:
            for ag in self.agents:
                ag.load_model(e=-1, model_path=self.network_model_path)

        self._run_attacked_episode(self.test_steps)
        self.logger.info(
            "Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d"
            % (
                self.metric.real_average_travel_time(),
                self.metric.rewards(),
                self.metric.queue(),
                self.metric.delay(),
                self.metric.throughput(),
            )
        )
        if self.wandb is not None:
            self.wandb.log(self._test_log_dict())
        elif self.comet is not None:
            self.comet.log_metrics(self._test_log_dict())
        return self.metric

    def _run_attacked_episode(
        self,
        num_steps,
        measure_decision_time=False,
        training=False,
        episode=0,
        total_decision_num=0,
        flush=0,
    ):
        self.metric.clear()
        last_obs = self.env.reset()
        dones = [False] * self.n_agents
        for ag in self.agents:
            ag.reset()

        get_time = time.process_time
        decision_time = 0.0
        episode_loss = []
        i = 0
        while i < num_steps:
            if i % self.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in self.agents])

                self._reset_fake_vehicles()
                self._inject_min_count_lanes()

                attacked_obs = [ag.get_ob() for ag in self.agents]
                if training and total_decision_num <= self.learning_start:
                    actions = np.stack([ag.sample() for ag in self.agents])
                    policy_obs = last_obs
                else:
                    actions = []
                    pre_decision_time = get_time()
                    for idx, ag in enumerate(self.agents):
                        actions.append(ag.get_action(attacked_obs[idx], last_phase[idx], test=not training))
                    if measure_decision_time:
                        decision_time += get_time() - pre_decision_time
                    actions = np.stack(actions)
                    policy_obs = attacked_obs

                actions_prob = []
                if training:
                    for idx, ag in enumerate(self.agents):
                        try:
                            actions_prob.append(ag.get_action_prob(policy_obs[idx], last_phase[idx]))
                        except Exception:
                            actions_prob.append(None)

                self._reset_fake_vehicles()

                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                    if i >= num_steps:
                        break
                rewards = np.mean(rewards_list, axis=0)
                self.metric.update(rewards)

                if training:
                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        done = dones[idx] if idx < len(dones) else all(dones)
                        ag.remember(
                            policy_obs[idx],
                            last_phase[idx],
                            actions[idx],
                            actions_prob[idx],
                            rewards[idx],
                            obs[idx],
                            cur_phase[idx],
                            done,
                            f"{episode}_{i // self.action_interval}_{ag.id}",
                        )

                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                    total_decision_num += 1
                    last_obs = obs

                if (
                    training
                    and total_decision_num > self.learning_start
                    and total_decision_num % self.update_model_rate == self.update_model_rate - 1
                ):
                    cur_loss_q = np.stack([ag.train() for ag in self.agents])
                    episode_loss.append(cur_loss_q)

                if (
                    training
                    and total_decision_num > self.learning_start
                    and total_decision_num % self.update_target_rate == self.update_target_rate - 1
                ):
                    [ag.update_target_network() for ag in self.agents]

            if all(dones):
                break

        self._reset_fake_vehicles()
        if measure_decision_time:
            return i, decision_time
        if training:
            return i, episode_loss, total_decision_num, flush
        return i

    def _inject_min_count_lanes(self):
        lane_counts = self._lane_counts()
        for inter in self.world.intersections:
            target = self._least_count_lane_target(inter, lane_counts)
            if target is None:
                continue

            approach, segment_idx = target
            vehicle_counts = [0] * max(3, segment_idx + 1)
            vehicle_counts[segment_idx] = self.fake_vehicle_budget
            self.world.inject_fake_vehicles(inter.id, approach, vehicle_counts)

        self._refresh_world()

    def _least_count_lane_target(self, inter, lane_counts):
        candidates = []
        for road in inter.in_roads:
            lanes = inter.road_lane_mapping.get(road, [])
            approach = self._road_to_approach(inter, road)
            if approach is None:
                continue

            for segment_idx, lane in enumerate(lanes[:3]):
                candidates.append(
                    (
                        self._safe_lane_count(lane_counts, lane),
                        str(lane),
                        approach,
                        segment_idx,
                    )
                )

        if not candidates:
            return None

        _, _, approach, segment_idx = min(candidates, key=lambda item: (item[0], item[1]))
        return approach, segment_idx

    def _road_to_approach(self, inter, road):
        for candidate_road, angle, is_out in zip(inter.roads, inter.directions, inter.outs):
            if candidate_road != road or is_out:
                continue
            if math.pi / 4 <= angle < 3 * math.pi / 4:
                return "N"
            if 3 * math.pi / 4 <= angle < 5 * math.pi / 4:
                return "W"
            if 5 * math.pi / 4 <= angle < 7 * math.pi / 4:
                return "S"
            return "E"
        return None

    def _lane_counts(self):
        try:
            return dict(self.world.get_info("lane_count"))
        except Exception:
            self.world.subscribe(["lane_count"])
            self._refresh_world()
            return dict(self.world.get_info("lane_count"))

    def _safe_lane_count(self, lane_counts, lane):
        value = lane_counts.get(lane, 0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0
        return 0 if math.isnan(value) else value

    def _reset_fake_vehicles(self):
        if hasattr(self.world, "reset_fake_vehicles"):
            self.world.reset_fake_vehicles()

    def _refresh_world(self):
        if hasattr(self.world, "_refresh_observations"):
            self.world._refresh_observations()
        elif hasattr(self.world, "_update_infos"):
            self.world._update_infos()

    def _controller_model_path(self):
        path = Registry.mapping["logger_mapping"]["path"].path
        for name in (
            "tsc_test_min_count_adversarial",
            "tsc_min_count_adversarial",
            "tsc_test",
        ):
            path = path.replace(name, "tsc")

        command_param = Registry.mapping["command_mapping"]["setting"].param
        target_network = command_param.get("network")
        controller_source_network = command_param.get("controller_source_network")
        if controller_source_network is not None and target_network is not None:
            path = path.replace(target_network, controller_source_network)
        return path

    def _test_log_dict(self):
        return {
            "Test/Travel Time": self.metric.real_average_travel_time(),
            "Test/Mean Reward": self.metric.rewards(),
            "Test/Mean Queue": self.metric.queue(),
            "Test/Mean Delay": self.metric.delay(),
            "Test/Throughput": self.metric.throughput(),
        }

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        res = (
            f"{Registry.mapping['model_mapping']['setting'].param['name']:<12}\t{mode:<8}\t{step:<6}\t"
            + f"{travel_time:<20}\t{loss:<20}\t{cur_rwd:<20}\t{cur_queue:<20}\t"
            + f"{cur_delay:<20}\t{cur_throughput:<20}"
        )
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()


@Registry.register_trainer("tsc_test_min_count_adversarial")
class TSCTesterMinCountAdversarial(TSCTrainerMinCountAdversarial):
    def test(self, drop_load=True):
        if Registry.mapping["command_mapping"]["setting"].param["world"] == "cityflow":
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, "final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()

        Registry.mapping["logger_mapping"]["path"].path = Registry.mapping["logger_mapping"]["path"].path.replace(
            "tsc_test", "tsc"
        )

        load_model = Registry.mapping["model_mapping"]["setting"].param.get("load_model")
        if load_model and load_model is not False:
            for ag in self.agents:
                ag.load_model(e=-1, model_path=self.network_model_path)

        for a in self.agents:
            a.reset()

        get_time = time.process_time
        pre_env_time = get_time()
        _, decision_time = self._run_attacked_episode(self.test_steps, measure_decision_time=True)
        env_time = get_time() - pre_env_time
        pct = decision_time / env_time * 100 if env_time > 0 else 0.0
        print(f"Simulation cost: {decision_time:.4f}/{env_time:.4f}|{pct:.4f}%")
        self.logger.info(
            "Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d"
            % (
                self.metric.real_average_travel_time(),
                self.metric.rewards(),
                self.metric.queue(),
                self.metric.delay(),
                self.metric.throughput(),
            )
        )
        return self.metric
