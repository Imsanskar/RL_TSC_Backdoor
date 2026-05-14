import os
import time

import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
from attacker.multi_ppo_attacker import MultiPPOAttacker


@Registry.register_trainer("tsc_max_adversarial")
class TSCTrainerMaxAdversarial(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc",
        wandb = None,
        comet = None
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later
        
        # TODO: support other dataset in the future
        # self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
        #     os.path.join(Registry.mapping['logger_mapping']['path'].path,
        #                  Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        # )
        # self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )
        
        self.wandb = wandb
        self.comet = comet

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))

        # for magd agents should share information 
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

        self.attacker_agents = []  # Each attacker agent controls one intersection (currently independent)

        # Handle both 'attacker' and 'attacker_mapping' for config compatibility
        attacker_config = Registry.mapping['attacker_mapping']['setting'].param
        if attacker_config.get('intersection', True):
            self.num_attacker_agent = len(self.world.intersections)
        else:
            self.num_attacker_agent = int(len(self.world.intersections) / agent.sub_agents)

        # Initialize Multi-PPO attacker for each intersection (from attacker/.py files)
        # Each attacker learns to inject fake vehicles to maximize victim's traffic delay

        # Get attacker parameters from config - use fallback values if not specified
        if Registry.mapping['command_mapping']['setting'].param['network'] == 'cityflow1x1':
             num_segments = 2
             num_approaches = 4
        else:
            num_segments = attacker_config.get('num_segments', 3)
            num_approaches = attacker_config.get('num_approaches', 4)
        learning_rate = attacker_config.get('learning_rate', 1e-4)
        gamma = attacker_config.get('gamma', 0.99)
        self.penalty_lambda = attacker_config.get('penalty_lambda', 0.01)
        max_vehicles_per_segment = attacker_config.get('max_vehicles_per_segment', 10)
        device = Registry.mapping['command_mapping']['setting'].param.get('device', 'cpu')

        self.max_vehicles_per_segment = max_vehicles_per_segment  # Store for use during testing
        for i in range(self.num_attacker_agent):
            attacker_kwargs = {
                'param': {
                    'learning_rate': float(learning_rate),
                    'gamma': float(gamma),
                    'penalty_lambda': float(self.penalty_lambda),
                    'max_vehicles_per_segment': float(max_vehicles_per_segment),
                    'num_segments': int(num_segments),
                    'num_approaches': int(num_approaches),
                    'device': device,
                }
            }
            self.attacker_agents.append(MultiPPOAttacker(self.world, i, **attacker_kwargs))

        # load model for controller, we are not training the controller

        self.n_agents = len(self.attacker_agents)  # Number of attacker agents (one per intersection or shared)

        model_path = Registry.mapping['logger_mapping']['path'].path.replace('tsc_max_adversarial', 'tsc')
        for ag in self.agents:
            ag.load_model(e = -1, model_path = model_path)


    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        # Initialize TSCEnv with attacker_agents parameter for adversarial training
        self.env = TSCEnv(
            self.world,
            self.agents,
            self.metric,
            attacker_agents=self.attacker_agents  # Pass attacker agents to environment
        )

    def train(self):
        pass

    def train_test(self, e):
        pass

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()

        model_path = Registry.mapping['logger_mapping']['path'].path.replace('tsc_max_adversarial', 'tsc')
        if not drop_load:
            [ag.load_model(self.episodes, model_path = model_path) for ag in self.agents]
        attention_mat_list = []
        obs = self.env.reset()
        dones = [False] * self.n_agents
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, _ in enumerate(self.attacker_agents):
                    vehicles_injected = 0
                    


                    # === Step 3: Inject fake vehicles if attacker selected an action ===
                    approaches = ['N', 'E', 'S', 'W']
                    approach_name = approaches[np.random.randint(0, 4)]  # Assuming 4 approaches per intersection
                    if Registry.mapping['attacker_mapping']['setting'].param['random']:
                        scale_action = np.random.randint(0, self.max_vehicles_per_segment + 1, (self.attacker_agents[idx].num_segments, ))  # Example: inject 10 vehicles in each segment of the approach
                    else:
                        scale_action = [self.max_vehicles_per_segment] * self.attacker_agents[idx].num_segments 
                    vehicle_counts = scale_action.tolist() if isinstance(scale_action, np.ndarray) else scale_action
                    vehicles_injected += self.world.inject_fake_vehicles(
                        self.attacker_agents[idx].intersection_id,
                        approach_name,
                        vehicle_counts
                    )

                obs = [agent.get_ob() for agent in self.agents]  # Get victim's observation after injection, which includes fake vehicles
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))

                self.world.reset_fake_vehicles()  # Remove fake vehicles before the controlled rollout
                actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (self.metric.real_average_travel_time(), \
            self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))
        
        if not self.wandb is None:
            self.wandb.log({
                'Test/Travel Time': self.metric.real_average_travel_time(),
                'Test/Mean Reward': self.metric.rewards(),
                'Test/Mean Queue': self.metric.queue(),
                'Test/Mean Delay': self.metric.delay(),
                'Test/Throughput': self.metric.throughput()
            })
        elif not self.comet is None:
            self.comet.log_metrics({
                'Test/Travel Time': self.metric.real_average_travel_time(),
                'Test/Mean Reward': self.metric.rewards(),
                'Test/Mean Queue': self.metric.queue(),
                'Test/Mean Delay': self.metric.delay(),
                'Test/Throughput': self.metric.throughput()
            })
        return self.metric

    def writeLog(self, mode, step, travel_time, critic_loss, actor_loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param critic_loss: current critic loss
        :param actor_loss: current actor loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        # res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
        #     step) + '\t' + "%.4f" % travel_time + '\t' + "%.4f" % loss + "\t" +\
        #     "%.4f" % cur_rwd + "\t" + "%.4f" % cur_queue + "\t" + "%.4f" % cur_delay + "\t" + "%d" % cur_throughput
        res = f"{Registry.mapping['model_mapping']['setting'].param['name']:<12}\t{mode:<8}\t{step:<6}\t"\
                + f"{travel_time:<20}\t{critic_loss:<20}\t{actor_loss:<20}\t{cur_rwd:<20}\t{cur_queue:<20}\t{cur_delay:<20}\t{cur_throughput:<20}"
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

@Registry.register_trainer("tsc_test_max_adversarial")
class TSCTesterMaxAdversarial(TSCTrainerMaxAdversarial):
    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()

        Registry.mapping['logger_mapping']['path'].path = Registry.mapping['logger_mapping']['path'].path.replace('tsc_test', 'tsc')
        # print(Registry.mapping['logger_mapping']['path'].path);exit()

        load_model = Registry.mapping['model_mapping']['setting'].param.get('load_model')
        if load_model and load_model is not False:
            for ag in self.agents:
                ag.load_model(self.episodes)
        attention_mat_list = []
        obs = self.env.reset()
        dones = [False] * self.n_agents
        for a in self.agents:
            a.reset()

        get_time = time.process_time
        pre_env_time = get_time()
        decision_time = 0.0
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []

                pre_decision_time = get_time()
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                decision_time += get_time() - pre_decision_time

                actions = np.stack(actions)
                rewards_list = []

                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))

                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        env_time = get_time() - pre_env_time
        print(f'Simulation cost: {decision_time:.4f}/{env_time:.4f}|{decision_time/env_time*100:.4f}%')
        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (self.metric.real_average_travel_time(), \
            self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))
        return self.metric