import os
import time

import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
from attacker.multi_ppo_attacker import MultiPPOAttacker


@Registry.register_trainer("tsc_rl_adversarial")
class TSCTrainerRLAdversarial(BaseTrainer):
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
        if 'cityflow1x1' in Registry.mapping['command_mapping']['setting'].param['network']:
             num_segments = 2
             num_approaches = 4
        else:
            num_segments = attacker_config.get('num_segments', 3)
            num_approaches = attacker_config.get('num_approaches', 4)
        actor_learning_rate = attacker_config.get('actor_learning_rate', 1e-4)
        critic_learning_rate = attacker_config.get('critic_learning_rate', 1e-4)
        gamma = attacker_config.get('gamma', 0.99)
        self.penalty_lambda = attacker_config.get('penalty_lambda', 0.01)
        max_vehicles_per_segment = attacker_config.get('max_vehicles_per_segment', 10)
        device = Registry.mapping['command_mapping']['setting'].param.get('device', 'cpu')

        for i in range(self.num_attacker_agent):
            attacker_kwargs = {
                'param': {
                    'actor_learning_rate': float(actor_learning_rate),
                    'critic_learning_rate': float(critic_learning_rate),
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

        if Registry.mapping['command_mapping']['setting'].param['task'] == 'tsc_rl_adversarial':
            self.network_model_path = Registry.mapping['logger_mapping']['path'].path.replace('tsc_rl_adversarial', 'tsc')
        elif Registry.mapping['command_mapping']['setting'].param['task'] == 'tsc_test_rl_adversarial':
            self.network_model_path = Registry.mapping['logger_mapping']['path'].path.replace('tsc_test_rl_adversarial', 'tsc')

        self.attacker_source_network = Registry.mapping['command_mapping']['setting'].param['attacker_source_network'] # traffic network used to train attacker model
        self.target_network = Registry.mapping['command_mapping']['setting'].param['network'] # traffic network used for training/evaluatoin of the attacker agent 
        self.controller_source_network = Registry.mapping['command_mapping']['setting'].param['controller_source_network'] # network used to train the controller
        
        if self.controller_source_network is not None:
            self.network_model_path = self.network_model_path.replace(self.target_network, self.controller_source_network)  # Load best model for testing; adjust as needed for training 

        for ag in self.agents:
            ag.load_model(e = -1, model_path = self.network_model_path)


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
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''
        total_decision_num = 0
        flush = 0
        max_travel_time = -float('inf')
        for e in range(self.episodes):
            # TODO: check this reset agent
            self.metric.clear()
            last_obs = self.env.reset()  # here the reset returns the initial observation for attacker agents 
            obs = last_obs
            dones_list = [False] * self.n_agents

            for ag in self.agents:
                ag.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            episode_loss = []
            i = 0

            episode_critic_losses = []
            episode_actor_losses = []
            
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    # === ADVERSARIAL ATTACK PHASE ===
                    # Attack flow from WIP paper:
                    # 1. Attacker observes state, selects fake vehicle injection action
                    # 2. Injects fake vehicles into simulation
                    # 3. Victim agent queries poisoned observation
                    # 4. Victim predicts phase based on poisoned observation
                    victim_actions = []

                    previous_attacker_state = [None] * len(self.attacker_agents)  # Initialize list to store previous states for each attacker agent
                    actions_prob = [None] * len(self.attacker_agents)  # Initialize list to store action probabilities for each attacker agent
                    values = [None] * len(self.attacker_agents)  # Initialize list to store value estimates for each attacker agent
                    vehicles_injected_list = [0] * len(self.attacker_agents)  # Initialize list to store number of vehicles injected by each attacker agent

                    before_attack_obs = [ag.get_ob() for ag in self.agents]  # Get observation before attack for replay buffer
                    for idx, att in enumerate(self.attacker_agents):
                        previous_attacker_state[idx] = att.get_state()  # Store previous state for replay buffer

                    for idx, ag in enumerate(self.attacker_agents):
                        if self.attacker_agents[idx] is not None:
                            # === Step 1: Attacker observes state ===
                        
                            att_state = previous_attacker_state[idx]  # Get current state for action selection

                            # === Step 2: Get attack action from Multi-PPO policy ===
                            (approach_action, scale_action), prob, value = self.attacker_agents[idx].get_action(att_state, test=False)

                            actions_prob[idx] = prob  # Store action probability for replay buffer
                            values[idx] = value  # Store value estimate for replay buffer
                            # === Step 3: Inject fake vehicles if attacker selected an action ===
                            if approach_action is not None and scale_action is not None:
                                approach_name = ['N', 'E', 'S', 'W'][approach_action % len(self.attacker_agents[idx]._approaches)]
                                vehicle_counts = np.asarray(scale_action, dtype=np.int32).tolist()
                                vehicles_injected = self.world.inject_fake_vehicles(
                                    self.attacker_agents[idx].intersection_id,
                                    approach_name,
                                    vehicle_counts
                                )
                                vehicles_injected_list[idx] = vehicles_injected
                            self.attacker_agents[idx].current_action = (approach_action, scale_action)


                    # === Step 5: Victim predicts phase based on (potentially poisoned) observation ===
                    for idx, ag in enumerate(self.agents):
                        victim_observation = ag.get_ob()
                        victim_action = self.agents[idx].get_action(victim_observation, last_phase[idx], test=True)
                        victim_actions.append(victim_action)

                    actions = np.stack(victim_actions) if len(victim_actions) > 0 else np.zeros_like(last_phase)
                    # Get action probabilities (victim agent only)

                    rewards_list = []

                    # Remove fake vehicles before the physical rollout so they only affect the victim decision.
                    self.world.reset_fake_vehicles()
                    
                    # === Execute action interval and collect transitions ===
                    for _ in range(self.action_interval):
                        obs, rewards, dones_list, infos = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))

                    # TODO: maximize the reward of attacker agent, which is the negative reward of controller agent
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)
                    rewards = rewards
                    if len(rewards.shape) == 1:
                        rewards = rewards.reshape((rewards.shape[0], -1))  # Ensure rewards is 2D [agent, intersection]
                    
                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])

                    # Store transitions for victim replay buffer
                    # for idx, ag in enumerate(self.attacker_agents):
                    #     ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx],
                    #         rewards[idx], obs[idx], cur_phase[idx], dones_list[idx],
                    #         f'{e}_{i//self.action_interval}_{ag.id}')

                    

                    # === Store attacker transitions in replay buffer ===
                    #TODO: Implement current phase as a state too
                    for idx, _ in enumerate(self.attacker_agents):
                        att = self.attacker_agents[idx]
                        approach_act = att.current_action[0] if hasattr(att, 'current_action') and att.current_action else 0
                        scale_act = att.current_action[1] if hasattr(att, 'current_action') and att.current_action else []
                        #attacker_reward = float(rewards[0][idx]) - self.penalty_lambda * vehicles_injected_list[idx]  # Example reward: negative of victim's reward minus penalty for number of vehicles injected
                        attacker_reward = float(rewards[0][0]) - self.penalty_lambda * vehicles_injected_list[idx]
                        att.observe(previous_attacker_state[idx], (approach_act, scale_act), (actions_prob[idx], values[idx]), attacker_reward, self.attacker_agents[idx].get_state(), dones_list[idx])


                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1
                    last_obs = obs

                
                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    cur_loss_q = np.stack([ag.update_policy() for ag in self.attacker_agents])

                    if not (None in cur_loss_q):
                        critic_losses = [loss['critic_loss'] for loss in cur_loss_q]
                        actor_losses = [loss['actor_loss'] for loss in cur_loss_q]
                        
                        #print(f"Episode {e}, Step {i}: Critic Loss: {np.mean(critic_losses):.4f}")
                        #print(f"Episode {e}, Step {i}: Actor Loss: {np.mean(actor_losses):.4f}")


                        episode_actor_losses.extend(actor_losses)
                        episode_critic_losses.extend(critic_losses)
                        # if self.comet is not None and len(critic_losses) > 0:
                        #     self.comet.log_metrics({
                        #         'Train/Mean Critic Loss': np.mean(critic_losses),
                        #         'Train/Mean Actor Loss': np.mean(actor_losses)
                        #     }, step=e)
                    
                # === Check episode termination when all done flags are True ===
                # Handle edge case where dones_list might be None
                done_flags = dones_list if dones_list is not None else [False] * self.n_agents
                if all(done_flags):
                    break
            if len(episode_critic_losses) > 0:
                mean_critic_loss = np.mean(np.array(episode_critic_losses))
            else:
                mean_critic_loss = 0

            if len(episode_actor_losses) > 0:
                mean_actor_loss = np.mean(np.array(episode_actor_losses))
            else:
                mean_actor_loss = 0

            self.writeLog("TRAIN", e, self.metric.real_average_travel_time(),\
                mean_critic_loss, mean_actor_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput())
            self.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps,\
                mean_critic_loss, mean_actor_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
            # if e % self.save_rate == 0:
            #     [ag.save_model(e=e) for ag in self.agents]
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes, self.metric.real_average_travel_time()))
            for j in range(len(self.world.intersections)):
                self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j],\
                     self.metric.lane_queue()[j]))
            # if self.test_when_train:
            metrics = {
                'Train/Travel Time': self.metric.real_average_travel_time(),
                'Train/Mean Critic Loss': mean_critic_loss,
                'Train/Mean Actor Loss': mean_actor_loss,
                'Train/Mean Reward': self.metric.rewards(),
                'Train/Mean Queue': self.metric.queue(),
                'Train/Mean Delay': self.metric.delay(),
                'Train/Throughput': self.metric.throughput()
            }


            real_travel_time = self.train_test(e)
            if real_travel_time > max_travel_time:
                max_travel_time = real_travel_time
                model_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
                [ag.save_model(model_path) for ag in self.attacker_agents]
                # [ag.save_model(e=self.episodes) for ag in self.agents]

            # if self.wandb is not None:
            #     self.wandb.log({
            #         **metrics,
            #         'Val/Travel Time': real_travel_time
            #     }, step=e)

            if self.comet is not None:
                self.comet.log_metrics({
                    **metrics,
                    'Val/Travel Time': real_travel_time
                }, step=e)


        #model_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model', 'last_0')
        model_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model', 'last_0','best_0.pth')
        [ag.save_model(model_path) for ag in self.attacker_agents]
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        # [ag.save_model(e=self.episodes) for ag in self.agents]

        

    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        obs = self.env.reset()
        dones = [False] * self.n_agents
        self.metric.clear()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                victim_actions = []

                ag = self.agents[0]  # Assuming single agent for simplicity; extend to multiple agents as needed
                for idx, _ in enumerate(self.attacker_agents):
                    vehicles_injected = 0
                    # === Step 1: Attacker observes state ===
                    state = self.attacker_agents[idx].get_state()

                    # === Step 2: Get attack action from Multi-PPO policy ===
                    (approach_action, scale_action), log_prob, value = self.attacker_agents[idx].get_action(
                        state, test=True
                    )

                    # === Step 3: Inject fake vehicles if attacker selected an action ===
                    vehicles_injected = 0
                    approach_name = ['N', 'E', 'S', 'W'][approach_action % len(self.attacker_agents[idx]._approaches)]
                    vehicle_counts = np.asarray(scale_action, dtype=np.int32).tolist()
                    vehicles_injected += self.world.inject_fake_vehicles(
                        self.attacker_agents[idx].intersection_id,
                        approach_name,
                        vehicle_counts
                    )


                    # Store attacker action for test phase replay buffer
                    if self.attacker_agents[idx] is not None and hasattr(self.attacker_agents[idx], 'current_action'):
                        self.attacker_agents[idx].current_action = (approach_action, scale_action)

                actions = []

                obs = [ag.get_ob() for ag in self.agents]  # Get new observation after potential attacker's injection
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))

                self.world.reset_fake_vehicles()  # Remove fake vehicles before the controlled rollout
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
                
            if all(dones):
                break
        self.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(\
            e, self.episodes, self.metric.real_average_travel_time(), self.metric.rewards(),\
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
        self.writeLog("TEST", e, self.metric.real_average_travel_time(),\
            100, 100, self.metric.rewards(),self.metric.queue(),self.metric.delay(), self.metric.throughput())
        
        return self.metric.real_average_travel_time()


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
        model_path = Registry.mapping['logger_mapping']['path'].path.replace('tsc_rl_adversarial', 'tsc')
        if not drop_load:
            [ag.load_model(self.episodes, model_path = model_path) for ag in self.agents]

        #model_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model', 'best_0')
        model_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model', 'best_0.pth')
        for ag in self.attacker_agents:
            if os.path.exists(model_path + '.pth'):
                model_path = model_path + '.pth' 
            ag.load_model(model_path)  # Load attacker model as well for testing
        attention_mat_list = []
        obs = self.env.reset()
        dones = [False] * self.n_agents
        for a in self.agents:
            a.reset()
        for a in self.attacker_agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, _ in enumerate(self.attacker_agents):
                    vehicles_injected = 0
                    
                    # === Step 1: Attacker observes state ===
                    state = self.attacker_agents[idx].get_state()

                    # === Step 2: Get attack action from Multi-PPO policy ===
                    (approach_action, scale_action), _, _ = self.attacker_agents[idx].get_action(
                        state, test=True
                    )

                    # === Step 3: Inject fake vehicles if attacker selected an action ===
                    if approach_action is not None and scale_action is not None:
                        approach_name = ['N', 'E', 'S', 'W'][approach_action % len(self.attacker_agents[idx]._approaches)]
                        vehicle_counts = scale_action.tolist() if isinstance(scale_action, np.ndarray) else scale_action
                        vehicles_injected = self.world.inject_fake_vehicles(
                            self.attacker_agents[idx].intersection_id,
                            approach_name,
                            vehicle_counts
                        )

                obs = [ag.get_ob() for ag in self.agents]  # Get new observation after potential attacker's injection
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

@Registry.register_trainer("tsc_test_rl_adversarial")
class TSCTesterRLAdversarial(TSCTrainerRLAdversarial):
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
        for ag in self.agents:
            ag.load_model(e = -1, model_path = self.network_model_path)

        model_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model', 'last_0','best_0.pth')
        for ag in self.attacker_agents:
            rank = ag.rank
            path = os.path.join(model_path, f'last_{rank}')
            if self.attacker_source_network is not None:
                model_path = model_path.replace(self.target_network, self.attacker_source_network)  # Load best model for testing; adjust as needed for training
            if os.path.exists(model_path + '.pth'):
                model_path = model_path + '.pth'
            ag.load_model(model_path)
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

                for idx, _ in enumerate(self.attacker_agents):
                    vehicles_injected = 0
                    
                    # === Step 1: Attacker observes state ===
                    state = self.attacker_agents[idx].get_state()

                    # === Step 2: Get attack action from Multi-PPO policy ===
                    (approach_action, scale_action), _, _ = self.attacker_agents[idx].get_action(
                        state, test=True
                    )

                    # === Step 3: Inject fake vehicles if attacker selected an action ===
                    if approach_action is not None and scale_action is not None:
                        approach_name = ['N', 'E', 'S', 'W'][approach_action % len(self.attacker_agents[idx]._approaches)]
                        vehicle_counts = scale_action.tolist() if isinstance(scale_action, np.ndarray) else scale_action
                        vehicles_injected += self.world.inject_fake_vehicles(
                            self.attacker_agents[idx].intersection_id,
                            approach_name,
                            vehicle_counts
                        )

                pre_decision_time = get_time()
                # obs = [ag.get_ob() for ag in self.agents]  # Get new observation after potential attacker's injection
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                decision_time += get_time() - pre_decision_time

                self.world.reset_fake_vehicles()  # Remove fake vehicles before the controlled rolloutq
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
