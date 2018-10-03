from algo.DQN import DQN
from explore.epsilon_greedy import ep_greedy
from explore.annealed_epsilon_greedy import annealed_ep_greedy
from memory.uniform_sampler import uniform_sampling
from memory.prioritized_sampler import prioritized_sampling
from network.q_network import q_network
from visualize.dumpData import dumpData
from collections import namedtuple
import torch.nn as nn
import torch


class RLExp():

    def __init__(self):
        self.avail_algorithms = ['DQN', 'DDQN']
        self.avail_policy = ['epsilon-greedy', 'annealing-ep-greedy']
        self.avail_memory = ['Uniform-Sampling', 'Prioritized-Sampling']

    def getAvailableAlgorithms(self):
        return self.avail_algorithms

    def getAvailableReplays(self):
        return self.avail_memory

    def getAvailableExplorePolicies(self):
        return self.avail_policy

    def setDevice(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device Set to : ",self.device)

    def setup_exp(self, algorithm, explore_policy, replay, observation_shape, num_actions, exp_name, save_interval):

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.time_steps_elapsed = 0
        self.save_interval = save_interval
        self.pack_sample = namedtuple("Sample",["prev_state","action","reward","next_state","done"])

        if algorithm not in self.avail_algorithms:
            raise ValueError("%s Algorithm is not implemented yet !" % (algorithm))
        else:
            self.algorithm = algorithm

        if explore_policy not in self.avail_policy:
            raise ValueError("%s is not implemented yet ! " % (explore_policy))
        else:
            self.explore_policy = explore_policy

        if replay not in self.avail_memory:
            raise ValueError("%s is not implemented yet !" % (replay))
        else:
            self.replay = replay

        if not len(str(exp_name)) > 0:
            raise ValueError("%s is an invalid experiment name"%(exp_name))
        else:
            self.exp_name = str(exp_name)

        # initialize Algorithm Class
        if self.algorithm == "DQN":
            self.algorithm_object = DQN()
        elif self.algorithm == "DDQN":
            self.algorithm_object = None

        # initialize exploration policy class
        if self.explore_policy == "epsilon-greedy":
            self.explore_policy_object = ep_greedy()
        elif self.explore_policy == "annealing-ep-greedy":
            self.explore_policy_object = annealed_ep_greedy()

        # initialize memory class
        if self.replay == "Uniform-Sampling":
            self.replay_object = uniform_sampling()
        elif self.replay == "Prioritized-Sampling":
            self.replay_object = prioritized_sampling()

        self.setDevice()
        # test provided network is correct or not by passing random input
        self.main_model = q_network(input_size=self.observation_shape, out_size= self.num_actions).to(self.device)
        self.target_model = q_network(input_size=self.observation_shape, out_size= self.num_actions).to(self.device)
        with torch.no_grad():
            random_input = torch.rand(2, self.observation_shape)
            self.main_model.forward(random_input, bsize=2)
            self.target_model.forward(random_input, bsize=2)

        # Copy weights initially
        self.target_model.load_state_dict(self.main_model.state_dict())

        print(self.main_model)

        # Policy setup
        self.explore_policy_object.actions = self.num_actions
        self.explore_policy_object.policy_net = self.main_model
        self.explore_policy_object.device = self.device

        # Algorithm setup
        self.algorithm_object.replay = self.replay_object
        self.algorithm_object.main_model = self.main_model
        self.algorithm_object.target_model = self.target_model
        self.algorithm_object.device = self.device

        # Dump Data Setup
        self.dumpData_object = dumpData(exp_name=self.exp_name, explore_policy=self.explore_policy_object, algorithm_object=self.algorithm_object)

        print("Algorithm Setup Done. Please set hyper-parameters.")
