import deap
import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import datetime

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import array
import random
import operator
import math
import os.path


from scoop import futures

#from novelty_search_vanila import *
import os
import torch



def to_numpy(var):
    return  var.data.numpy()



class Genome(nn.Module):

    def __init__(self, **kwargs):
        super(Genome, self).__init__()
        self.state_dim = kwargs["nb_input"]
        self.action_dim = kwargs["nb_output"]
        self.max_action = kwargs["max_action"]

    def set_parameters(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_parameters(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_parameters().shape[0]

    def load_model(self, filename, net_name):
        """
        Loads the model
        """
        if filename is None:
            return
        super(RLNN, self).load_model(filename=filename)

    def save_model(self, output, net_name):
        """
        Saves the model
        """
        super(RLNN, self).save_model(filename=output)



class Regression(Genome):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.max_action = kwargs["max_action"]
        self.l1 = nn.Linear(kwargs["nb_input"],10)
        self.l2 = nn.Linear(10,10)
        self.l3 = nn.Linear(10,10)
        self.out = nn.Linear(10,kwargs["nb_output"])

               
    def predict(self, x):
        x = torch.tensor(x)
        x = self.sigm(self.l1(x))
        x = self.sigm(self.l2(x))
        x = self.sigm(self.l3(x))

        x = self.out(x)
        x = torch.tanh(x)
        x = x.detach().numpy()
        return x*self.max_action
