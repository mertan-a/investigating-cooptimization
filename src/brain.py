import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
from copy import deepcopy
import _pickle as pickle

from networks import NeuralNetwork

class BRAIN(object):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def mutate(self):
        raise NotImplementedError

    @staticmethod
    def name(self):
        return self.name

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def is_valid(self):
        raise NotImplementedError

    def extract_brain(self):
        raise NotImplementedError

    def get_action(self, observation):
        # get the actions
        actions = self.model.forward(torch.from_numpy(observation).double())
        # turn the actions into numpy array
        actions = actions.detach().numpy()
        return actions

    def update_model(self):
        if hasattr(self, 'weight_list'):
            self.vector_to_weight_list()
            for idx, w in enumerate(self.weight_list):
                vector_to_parameters(w, self.model_list[idx].parameters())
                self.model_list[idx].double()
        else:
            if type(self.weights) != torch.Tensor:
                self.weights = torch.from_numpy(self.weights).double()
            vector_to_parameters(self.weights, self.model.parameters())
            self.model.double()


class CENTRALIZED(BRAIN):
    '''
    '''
    def __init__(self, args):
        BRAIN.__init__(self, "CENTRALIZED", args)
        self.mu, self.sigma = 0, 0.1
        if self.args.use_pretrained_brain:
            self.model = self.extract_brain(self.args.pretrained_brain)
            self.weights = parameters_to_vector(self.model.parameters())
        else:
            # input size depends on bounding box
            input_size = 0
            input_size += 5 * self.args.bounding_box[0] * self.args.bounding_box[1] # structure
            input_size += 1 # time
            input_size += 2 * self.args.bounding_box[0] * self.args.bounding_box[1] # velocity
            input_size += 1 * self.args.bounding_box[0] * self.args.bounding_box[1] # volume

            output_size = self.args.bounding_box[0] * self.args.bounding_box[1]

            self.model = NeuralNetwork(input_size, output_size)
            for p in self.model.parameters():
                p.requires_grad = False
            self.weights = parameters_to_vector(self.model.parameters())
        self.model.double()
        self.model.eval()

    def mutate(self):
        noise_weights = np.random.normal(self.mu, self.sigma,
                                         self.weights.shape)
        self.weights += noise_weights
        self.update_model()
        return noise_weights

    def extract_brain(self, path_to_pkl):
        # path should be a .pkl file
        with open(path_to_pkl, 'rb') as f:
            population = pickle.load(f)
        return population[0].brain.model

    def is_valid(self):
        return True


class DECENTRALIZED(BRAIN):
    '''
    '''
    def __init__(self, args):
        BRAIN.__init__(self, "DECENTRALIZED", args)
        self.mu, self.sigma = 0, 0.1
        if self.args.use_pretrained_brain:
            self.model = self.extract_brain(self.args.pretrained_brain)
            self.weights = parameters_to_vector(self.model.parameters())
        else:
            input_size = 0
            input_size += 5 * (3)**2 # structure
            input_size += 1 * (3)**2 # volume
            input_size += 1 # time
            input_size += 2 * (3)**2 # velocity

            output_size = 1

            self.model = NeuralNetwork(input_size, output_size)
            for p in self.model.parameters():
                p.requires_grad = False
            self.weights = parameters_to_vector(self.model.parameters())
        self.model.double()
        self.model.eval()

    def mutate(self):
        noise_weights = np.random.normal(self.mu, self.sigma,
                                         self.weights.shape)
        self.weights += noise_weights
        vector_to_parameters(self.weights, self.model.parameters())
        self.model.double()
        return noise_weights

    def extract_brain(self, path_to_pkl):
        # path should be a .pkl file
        with open(path_to_pkl, 'rb') as f:
            population = pickle.load(f)
        return population[0].brain.model

    def is_valid(self):
        return True


class OPEN_LOOP(BRAIN):
    '''
    open loop controller
    uses global observation wrapper
    pulls the time from the global observation
    and the action is the same as the time
    '''
    def __init__(self, args):
        BRAIN.__init__(self, "OPEN_LOOP", args)

    def get_action(self, observation):
        actions = np.ones((self.args.bounding_box[0]*self.args.bounding_box[1])) * observation[-1] - 0.4
        return actions

    def mutate(self):
        return True

    def is_valid(self):
        return True

