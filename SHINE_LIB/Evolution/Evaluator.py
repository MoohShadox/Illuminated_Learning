import deap
import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from scipy.spatial import KDTree

import datetime
import MultiNEAT as NEAT

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


class Behavior:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    

    def __getitem__(self, i):
        return self.x if i == 0 else self.y

    def distance_to(self, other):
        return np.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)


def Neuro_Evolve_Evaluation(genome, resdir = "res", render=False, dump=False, name="", **kwargs):
    """ Evaluation of a neural network. Returns the fitness, the behavior descriptor and a log of what happened
        Consider using dump=True to generate log files. These files are put in the resdir directory.
    """
    env = gym.make(kwargs['gym_name'], **kwargs['env_params'])
    nbstep=kwargs["episode_nb_step"]
    
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    
    observation = env.reset()
    observation, reward, done, info = env.step([0]*kwargs["nb_output"])

    #print("First observation: "+str(observation)+" first pos: "+str(env.get_robot_pos()))
    if (dump):
        f={}
        for k in info.keys():
            fn=resdir+"/traj_"+k+"_"+name+".log"
            if (os.path.exists(fn)):
                cpt=1
                fn=resdir+"/traj_"+k+"_"+name+"_%d.log"%(cpt)
                while (os.path.exists(fn)):
                    cpt+=1
                    fn=resdir+"/traj_"+k+"_"+name+"_%d.log"%(cpt)
            f[k]=open(fn,"w")

    action_scale_factor = env.action_space.high
    episode_reward=0
    episode_bd=None
    episode_log={}
    for t in range(nbstep):
        if render:
            env.render()
        
        net.Input(observation)
        net.Activate()
        action = 2*np.array(net.Output())
        #print("Observation: "+str(observation)+" Action: "+str(action))
        observation, reward, done, info = env.step(action) 
        if (kwargs["episode_reward_kind"] == "cumul"):
            episode_reward+=reward

        for k in kwargs["episode_log"].keys():
            if (kwargs["episode_log"][k] == "cumul"):
                if (k not in episode_log.keys()):
                    episode_log[k] = info[k]
                else:
                    episode_log[k] += info[k]
        if(dump):
            for k in f.keys():
                if (isinstance(info[k], list) or isinstance(info[k], tuple)):
                    data=" ".join(map(str,info[k]))
                else:
                    data=str(info[k])
                f[k].write(data+"\n")
        if(done):
            break
    if (dump):
        for k in f.keys():
            f[k].close()

    if (kwargs["episode_reward_kind"] == "final"):
        episode_reward=reward
        
    if (kwargs["episode_bd_kind"] == "final"):
        episode_bd=info[kwargs["episode_bd"]][slice(*kwargs["episode_bd_slice"])]
        
    for k in kwargs["episode_log"].keys():
        if (kwargs["episode_log"][k] == "final"):
            episode_log[k] = info[k]
    
    if(episode_log["exit_reached"]==1.0):
            print("Target REACHED ! ")
    #print("End of eval,  total_dist=%f"%(episode_reward))
    episode_reward = -episode_reward
    episode_reward = np.sqrt(kwargs["grid_max_v"][0]**2 + kwargs["grid_max_v"][0]**2) - episode_reward
    #print("REWARD : ",episode_reward)
    return episode_reward, Behavior(episode_bd[0],episode_bd[1]) , episode_log


def Simple_Gene_Evaluator(genotype,nn_class, resdir, render=False, dump=False, name="", **kwargs):
    """ Evaluation of a neural network. Returns the fitness, the behavior descriptor and a log of what happened
        Consider using dump=True to generate log files. These files are put in the resdir directory.
    """
    genotype = np.array(genotype)
    env = gym.make(kwargs['gym_name'], **kwargs['env_params'])
    nbstep=kwargs["episode_nb_step"]
    nn=nn_class(**kwargs)
    nn.set_parameters(genotype)
    observation = env.reset()
    observation, reward, done, info = env.step([0]*kwargs["nb_output"])

    #print("First observation: "+str(observation)+" first pos: "+str(env.get_robot_pos()))
    if (dump):
        f={}
        for k in info.keys():
            fn=resdir+"/traj_"+k+"_"+name+".log"
            if (os.path.exists(fn)):
                cpt=1
                fn=resdir+"/traj_"+k+"_"+name+"_%d.log"%(cpt)
                while (os.path.exists(fn)):
                    cpt+=1
                    fn=resdir+"/traj_"+k+"_"+name+"_%d.log"%(cpt)
            f[k]=open(fn,"w")

    action_scale_factor = env.action_space.high
    episode_reward=0
    episode_bd=None
    episode_log={}
    for t in range(nbstep):
        if render:
            env.render()
        action=nn.predict(observation)
        action=action_scale_factor*np.array(action)
        #print("Observation: "+str(observation)+" Action: "+str(action))
        observation, reward, done, info = env.step(action) 
        if (kwargs["episode_reward_kind"] == "cumul"):
            episode_reward+=reward

        for k in kwargs["episode_log"].keys():
            if (kwargs["episode_log"][k] == "cumul"):
                if (k not in episode_log.keys()):
                    episode_log[k] = info[k]
                else:
                    episode_log[k] += info[k]
        if(dump):
            for k in f.keys():
                if (isinstance(info[k], list) or isinstance(info[k], tuple)):
                    data=" ".join(map(str,info[k]))
                else:
                    data=str(info[k])
                f[k].write(data+"\n")
        if(done):
            break
    if (dump):
        for k in f.keys():
            f[k].close()

    if (kwargs["episode_reward_kind"] == "final"):
        episode_reward=reward
        
    if (kwargs["episode_bd_kind"] == "final"):
        episode_bd=info[kwargs["episode_bd"]][slice(*kwargs["episode_bd_slice"])]
        
    for k in kwargs["episode_log"].keys():
        if (kwargs["episode_log"][k] == "final"):
            episode_log[k] = info[k]
    
    if(episode_log["exit_reached"]==1.0):
            print("Target REACHED ! ")
    #print("End of eval, t=%d, total_dist=%f"%(t,total_dist))
    return episode_reward, episode_bd, episode_log