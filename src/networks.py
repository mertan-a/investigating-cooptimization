import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import sigmoid
import numpy as np
from numba import njit
from copy import deepcopy

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return sigmoid(logits) - 0.4


class HebbianNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(HebbianNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, self.output_size, bias=False)
        )
        # initialize the weights in relu_stack from a uniform distribution
        self.reset_weights()
        # get the weights of the network
        self.w1, self.w2 = list(self.parameters())
        # TODO: for now, we assume ABCD hebbian learning with 4 parameters per synapse
        self.hebbian_type = 'ABCD'
        self.hebbian_params = 4
        # randomly initialize the hebbian parameters
        self.hebbian_params = np.random.uniform(-1,1,(np.prod(self.w1.shape)+np.prod(self.w2.shape), self.hebbian_params))

    def forward(self, x):
        # save the input as a numpy array
        inp = x.detach().numpy()
        # forward pass, save intermediate activations
        act = self.relu_stack[1](self.relu_stack[0](x))
        out = sigmoid(self.relu_stack[2](act)) - 0.4
        # turn intermediate results to numpy
        a = act.detach().numpy()
        o = out.detach().numpy()
        # reshape the inp, a, o to be 2D with the first dimension being the batch size
        if len(inp.shape) == 1:
            inp = inp.reshape(1, -1)
            a = a.reshape(1, -1)
            o = o.reshape(1, -1)
        # update the hebbian parameters
        hebbian_update_ABCD(self.hebbian_params, self.w1.detach().numpy(), self.w2.detach().numpy(), inp, a, o)
        # normalize the weights of the network
        self.normalize_weights()
        return out

    def reset_weights(self):
        for m in self.relu_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1, 1)

    def normalize_weights(self):
        # normalize the weights of the network
        self.w1.data = self.w1.data / torch.norm(self.w1.data, dim=1, keepdim=True)
        self.w2.data = self.w2.data / torch.norm(self.w2.data, dim=1, keepdim=True)

# adapted from https://github.com/enajx/HebbianMetaLearning
@njit
def hebbian_update_ABCD(heb_coeffs, weights1_2, weights2_3, o0, o1, o2):
    heb_offset = 0
    # Layer 1
    for b in range(o0.shape[0]):
        for i in range(weights1_2.shape[1]):
            for j in range(weights1_2.shape[0]):
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] + ( heb_coeffs[idx][0] * o0[b][i] * o1[b][j]
                                                           + heb_coeffs[idx][1] * o0[b][i]
                                                           + heb_coeffs[idx][2] * o1[b][j])

    heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
    # Layer 2
    for b in range(o0.shape[0]):
        for i in range(weights2_3.shape[1]):
            for j in range(weights2_3.shape[0]):
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] + ( heb_coeffs[idx][0] * o1[b][i] * o2[b][j]
                                                           + heb_coeffs[idx][1] * o1[b][i]
                                                           + heb_coeffs[idx][2] * o2[b][j])

    return weights1_2, weights2_3

# test code
if __name__ == '__main__':
    # code for getting the model sizes for different controllers
    mlp_5x5 = NeuralNetwork(input_size=25*5+25*2+25*1+1, output_size=25)
    print('mlp_5x5', parameters_to_vector(mlp_5x5.parameters()).shape)
    mlp_7x7 = NeuralNetwork(input_size=49*5+49*2+49*1+1, output_size=49)
    print('mlp_7x7', parameters_to_vector(mlp_7x7.parameters()).shape)
    mlp_modular_or1 = NeuralNetwork(input_size=9*5+9*2+9*1+1, output_size=1)
    print('mlp_modular_or1', parameters_to_vector(mlp_modular_or1.parameters()).shape)
    exit()

    mlp = NeuralNetwork(input_size=10, output_size=25)
    inp = torch.rand(3,10)
    out = mlp(inp)
    print(out.shape)
    hebbian_mlp = HebbianNeuralNetwork(input_size=10, output_size=25)
    hebbian_out = hebbian_mlp(inp)
    print(hebbian_out.shape)





