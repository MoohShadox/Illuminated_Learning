#################################################################################################
#                                                                                               #
#                                                                                               #
#                Evolutionary playground: from convergent to divergent search                   #
#                                                                                               #
#                                                                                               #
#################################################################################################
#                                                                                               #
#                                                                                               #
#   Copyright (C) 2020 Stephane Doncieux, Sorbonne Université                                   #
#                                                                                               #
#  This program is free software; you can redistribute it and/or modify it under the terms      #
#  of the GNU General Public License as published by the Free Software Foundation;              #
#  either version 2 of the License, or (at your option) any later version.                      #
#                                                                                               #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;    #
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    #
#  See the GNU General Public License for more details.                                         #
#                                                                                               #
#  You should have received a copy of the GNU General Public License along with this program;   #
#  if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,             #
#  Boston, MA 02111-1307 USA                                                                    #
#                                                                                               #
#                                                                                               #
#################################################################################################
#                                                                                               #
# This code allows to run different variants of gradient free direct policy search algorithms   #
# It relies on the DEAP framework to allow an easy exploration of EA components (selection,     #
# mutation, ...), see https://deap.readthedocs.io for more details.                             #
#                                                                                               #
# To use it, set the env_name variable below and launch it with python:                                                        #
#       python3 ea_dps.py                                                                        #
#                                                                                               #
# If you have multiple cores on your computer, consider using scoop, it will parallelize        #
# the run and thus greatly accelerate it:                                                       #
#       python3 -m scoop ea_dps.py                                                               #
#                                                                                               #
#################################################################################################



import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
from scipy.spatial import KDTree

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

from plot import *

from scoop import futures

#from novelty_search_vanila import *
from novelty_search import *
import os

import grid_management

registered_envs={}


# all parameters of a run are here, change it or duplicate it to explore different possibilities
# the parameters are grouped into subset of parameters to limite their duplication.
# They are concatenated below to create registered_envs entries.

fastsim_env1={
    'gym_name': 'FastsimSimpleNavigation-v0',
    'env_params': {"still_limit": 10, 'reward_kind': "continuous"},

    'nb_input': 5, # number of NN inputs
    'nb_output': 2, # number of NN outputs
    'nb_layers': 2, # number of layers
    'nb_neurons_per_layer': 10, # number of neurons per layer

    'episode_nb_step': 1000, # maximum number of steps during an episode
    'episode_reward_kind': 'final', # 2 possible values: 'final' (the reward of an episode is the last observed reward and 'cumul' (the reward of an episode is the sum of all observed rewards
    'episode_bd': 'robot_pos', # the info key value to use as a bd
    'episode_bd_slice': (0,2,None), # specify the slice of the BD you are interested in (start, stop, step), see slice function. put (None, None, None) if you want everything
    'episode_bd_kind': 'final', # only final for the moment
    'episode_log': {'collision': 'cumul', 
                    'dist_obj': 'final', 
                    'exit_reached': 'final', 
                    'robot_pos': 'final'},

    'dim_grid': [100, 100],
    'grid_min_v': [0,0],
    'grid_max_v': [600,600],
    'goal': [60,60],

    'watch_max': 'dist_obj', # watching for the max in the corresponding info entry
}

fastsim_env2={
    'gym_name': 'FastsimSimpleNavigation-v0',
    'env_params': {"still_limit": 10, 'reward_kind': "collisions"},

    'nb_input': 5, # number of NN inputs
    'nb_output': 2, # number of NN outputs
    'nb_layers': 2, # number of layers
    'nb_neurons_per_layer': 10, # number of neurons per layer

    'episode_nb_step': 1000, # maximum number of steps during an episode
    'episode_reward_kind': 'cumul', # 2 possible values: 'final' (the reward of an episode is the last observed reward and 'cumul' (the reward of an episode is the sum of all observed rewards
    'episode_bd': 'robot_pos', # the info key value to use as a bd
    'episode_bd_slice': (0,2,None), # specify the slice of the BD you are interested in (start, stop, step), see slice function. put (None, None, None) if you want everything
    'episode_bd_kind': 'final', # only final for the moment
    'episode_log': {'collision': 'cumul', 
                    'dist_obj': 'final', 
                    'exit_reached': 'final', 
                    'robot_pos': 'final'},

    'dim_grid': [100, 100],
    'grid_min_v': [0,0],
    'grid_max_v': [600,600],
    'goal': [60,60],
    'watch_max': 'dist_obj', # watching for the max in the corresponding info entry

}

ea_generic={
    'min_value': -30, # min genotype value
    'max_value': 30, # max genotype value
    'min_strategy': 0.5, # min value for the mutation
    'max_strategy': 3, # max value for the mutation
    'nb_gen': 25, # number of generations
    'mu': 100, # population size
    'lambda': 200, # number of individuals generated
    'nov_k': 15, # k parameter of novelty search
    'nov_add_strategy': "random", # archive addition strategy (either 'random' or 'novel')
    'nov_lambda': 6, # number of individuals added to the archive
}

ea_random_sampling={
    'min_value': -30, # min genotype value
    'max_value': 30, # max genotype value
    'min_strategy': 0.5, # min value for the mutation
    'max_strategy': 3, # max value for the mutation
    'nb_gen': 25, # number of generations
    'mu': 200100, # population size
    'lambda': 200, # number of individuals generated
    'nov_k': 15, # k parameter of novelty search
    'nov_add_strategy': "random", # archive addition strategy (either 'random' or 'novel')
    'nov_lambda': 6, # number of individuals added to the archive
}


ea_NS={
    'selection': 'NS', # can be either NS, FIT or FIT+NS
}

ea_FIT={
    'selection': 'FIT', # can be either NS, FIT or FIT+NS
}

ea_FIT_NS={
    'selection': 'FIT+NS', # can be either NS, FIT or FIT+NS
}



# Fastsim with NS
registered_envs["fastsim_NS"]={}
registered_envs["fastsim_NS"].update(fastsim_env1)
registered_envs["fastsim_NS"].update(ea_generic)
registered_envs["fastsim_NS"].update(ea_NS)


# Fastsim with FIT
registered_envs["fastsim_FIT"]={}
registered_envs["fastsim_FIT"].update(fastsim_env1)
registered_envs["fastsim_FIT"].update(ea_generic)
registered_envs["fastsim_FIT"].update(ea_FIT)

# Fastsim with FIT and NS (multi-objective approach)
registered_envs["fastsim_FIT_NS"]={}
registered_envs["fastsim_FIT_NS"].update(fastsim_env1)
registered_envs["fastsim_FIT_NS"].update(ea_generic)
registered_envs["fastsim_FIT_NS"].update(ea_FIT_NS)

# Fastsim with random sampling
registered_envs["fastsim_RANDOM"]={}
registered_envs["fastsim_RANDOM"].update(fastsim_env1)
registered_envs["fastsim_RANDOM"].update(ea_random_sampling)
registered_envs["fastsim_RANDOM"].update(ea_FIT_NS)



# change this variable to choose the environment you are interested in 
# (one among the keys of registered_envs)
env_name="fastsim_FIT_NS"


def eval_nn(genotype, resdir, render=False, dump=False, name=""):
    """ Evaluation of a neural network. Returns the fitness, the behavior descriptor and a log of what happened
        Consider using dump=True to generate log files. These files are put in the resdir directory.
    """
    nbstep=registered_envs[env_name]["episode_nb_step"]
    nn=SimpleNeuralControllerNumpy(registered_envs[env_name]["nb_input"],
                                   registered_envs[env_name]["nb_output"],
                                   registered_envs[env_name]["nb_layers"],
                                   registered_envs[env_name]["nb_neurons_per_layer"])
    nn.set_parameters(genotype)
    observation = env.reset()
    observation, reward, done, info = env.step([0]*registered_envs[env_name]["nb_output"]) # if we forget that, the initial perception may be different from one eval to another... 
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
        if (registered_envs[env_name]["episode_reward_kind"] == "cumul"):
            episode_reward+=reward

        for k in registered_envs[env_name]["episode_log"].keys():
            if (registered_envs[env_name]["episode_log"][k] == "cumul"):
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

    if (registered_envs[env_name]["episode_reward_kind"] == "final"):
        episode_reward=reward
        
    if (registered_envs[env_name]["episode_bd_kind"] == "final"):
        episode_bd=info[registered_envs[env_name]["episode_bd"]][slice(*registered_envs[env_name]["episode_bd_slice"])]
        
    for k in registered_envs[env_name]["episode_log"].keys():
        if (registered_envs[env_name]["episode_log"][k] == "final"):
            episode_log[k] = info[k]
    
    #print("End of eval, t=%d, total_dist=%f"%(t,total_dist))
    return episode_reward, episode_bd, episode_log

nn=SimpleNeuralControllerNumpy(registered_envs[env_name]["nb_input"],
                               registered_envs[env_name]["nb_output"],
                               registered_envs[env_name]["nb_layers"],
                               registered_envs[env_name]["nb_neurons_per_layer"])
center=nn.get_parameters()


if (registered_envs[env_name]['selection']=="FIT+NS"):
    creator.create("MyFitness", base.Fitness, weights=(1.0,1.0))
elif (registered_envs[env_name]['selection']=="FIT"):
    creator.create("MyFitness", base.Fitness, weights=(1.0,))
elif (registered_envs[env_name]['selection']=="NS"):
    creator.create("MyFitness", base.Fitness, weights=(1.0,))
elif (registered_envs[env_name]['selection']=="NSLC"):
    creator.create("MyFitness", base.Fitness, weights=(1.0,1.0))
else:
    print("Variante inconnue: "+registered_envs[env_name]['selection'])

creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

IND_SIZE=len(center)



grid={}


def launch_ea(mu=100, lambda_=200, cxpb=0.3, mutpb=0.7, ngen=100, verbose=False, resdir="res"):

    random.seed()

    # Preparation of the EA with the DEAP framework. See https://deap.readthedocs.io for more details.
    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy, IND_SIZE, 
                     registered_envs[env_name]["min_value"], 
                     registered_envs[env_name]["max_value"], 
                     registered_envs[env_name]["min_strategy"], 
                     registered_envs[env_name]["max_strategy"])

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
    toolbox.register("select", tools.selNSGA2)
    
    toolbox.register("map",futures.map)
    toolbox.decorate("mate", checkStrategy(registered_envs[env_name]["min_strategy"]))
    toolbox.decorate("mutate", checkStrategy(registered_envs[env_name]["min_strategy"]))
    toolbox.register("evaluate", eval_nn, resdir=resdir)


    population = toolbox.population(n=mu)
    paretofront = tools.ParetoFront()
    
    fbd=open(resdir+"/bd.log","w")
    finfo=open(resdir+"/info.log","w")
    ffit=open(resdir+"/fit.log","w")

    nb_eval=0

    ##
    ### Initial random generation: beginning
    ##

    # Evaluate the individuals with an invalid (i.e. not yet evaluated) fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses_bds = toolbox.map(toolbox.evaluate, invalid_ind)
    nb_eval+=len(invalid_ind)
    finfo.write("## Generation 0 \n")
    finfo.flush()
    ffit.write("## Generation 0 \n")
    ffit.flush()
    for ind, (fit, bd, log) in zip(invalid_ind, fitnesses_bds):
        #print("Fit: "+str(fit)) 
        #print("BD: "+str(bd))
        
        if (registered_envs[env_name]['selection']=="FIT+NS"):
            ind.fitness.values=(fit,0)
        elif (registered_envs[env_name]['selection']=="FIT"):
            ind.fitness.values=(fit,)
        elif (registered_envs[env_name]['selection']=="NS"):
            ind.fitness.values=(0,)
        elif (registered_envs[env_name]['selection']=="NSLC"):
            ind.fitness.values=(0,0)
        
        ind.fit = fit
        ind.log = log
        ind.bd = bd
        fbd.write(" ".join(map(str,bd))+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(fit)+"\n")
        ffit.flush()
        grid_management.add_to_grid(grid,ind, fit, 
                                    dim=registered_envs[env_name]['dim_grid'], 
                                    min_v=registered_envs[env_name]['grid_min_v'], 
                                    max_v=registered_envs[env_name]['grid_max_v'])
        
    if paretofront is not None:
        paretofront.update(population)


    archive=updateNovelty(population,population,None,
                          registered_envs[env_name]['nov_k'],
                          registered_envs[env_name]['nov_add_strategy'],
                          registered_envs[env_name]['nov_lambda'])

    for ind in population:
        if (registered_envs[env_name]['selection']=="FIT+NS"):
            ind.fitness.values=(ind.fit,ind.novelty)
        elif (registered_envs[env_name]['selection']=="FIT"):
            ind.fitness.values=(ind.fit,)
        elif (registered_envs[env_name]['selection']=="NS"):
            ind.fitness.values=(ind.novelty,)

        #print("Fit=%f Nov=%f"%(ind.fit, ind.novelty))

    indexmax, valuemax = max(enumerate([i.log[registered_envs[env_name]['watch_max']] for i in population]), key=operator.itemgetter(1))

    ##
    ### Initial random generation: end
    ##


    # Begin the generational process
    for gen in range(1, ngen + 1):
        finfo.write("## Generation %d \n"%(gen))
        finfo.flush()
        ffit.write("## Generation %d \n"%(gen))
        ffit.flush()
        if (gen%10==0):
            print("+",end="", flush=True)
        else:
            print(".",end="", flush=True)

        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)


        # Evaluate the individuals with an invalid (i.e. not yet evaluated) fitness
        invalid_ind = [ind for ind in offspring]
        fitnesses_bds = toolbox.map(toolbox.evaluate, invalid_ind)
        nb_eval+=len(invalid_ind)

        for ind, (fit, bd, log) in zip(invalid_ind, fitnesses_bds):
            #print("Fit: "+str(fit)+" BD: "+str(bd)) 
            if (registered_envs[env_name]['selection']=="FIT+NS"):
                ind.fitness.values=(fit,0)
            elif (registered_envs[env_name]['selection']=="FIT"):
                ind.fitness.values=(fit,)
            elif (registered_envs[env_name]['selection']=="NS"):
                ind.fitness.values=(0,)
            elif (registered_envs[env_name]['selection']=="NSLC"):
                ind.fitness.values=(0,0)
            ind.fit = fit
            ind.bd = bd
            ind.log=log
            fbd.write(" ".join(map(str,bd))+"\n")
            fbd.flush()
            finfo.write(str(log)+"\n")
            finfo.flush()
            ffit.write(str(fit)+"\n")
            ffit.flush()

            grid_management.add_to_grid(grid,ind, ind.fit, 
                                        dim=registered_envs[env_name]['dim_grid'], 
                                        min_v=registered_envs[env_name]['grid_min_v'], 
                                        max_v=registered_envs[env_name]['grid_max_v'])


        pq=population+offspring

        archive=updateNovelty(pq,offspring,archive,
                              registered_envs[env_name]['nov_k'],
                              registered_envs[env_name]['nov_add_strategy'],
                              registered_envs[env_name]['nov_lambda'])

        for ind in pq:
            if (registered_envs[env_name]['selection']=="FIT+NS"):
                ind.fitness.values=(ind.fit,ind.novelty)
            elif (registered_envs[env_name]['selection']=="FIT"):
                ind.fitness.values=(ind.fit,)
            elif (registered_envs[env_name]['selection']=="NS"):
                ind.fitness.values=(ind.novelty,)

            #print("Fitness values: "+str(ind.fitness.values)+" Fit=%f Nov=%f"%(ind.fit, ind.novelty))
        

        # Select the next generation population
        population[:] = toolbox.select(pq, mu)

        

        indexmax, newvaluemax = max(enumerate([i.log[registered_envs[env_name]['watch_max']] for i in pq]), key=operator.itemgetter(1))
        if (newvaluemax>valuemax):
            valuemax=newvaluemax
            print("Gen "+str(gen)+", new max ! max fit="+str(valuemax)+" index="+str(indexmax)+" BD="+str(pq[indexmax].bd))
            nnfit, nnbd, log = eval_nn(pq[indexmax],resdir,render=True,dump=True,name="gen%04d"%(gen))
    fbd.close()
    finfo.close()
    ffit.close()

    
    grid_management.stat_grid(grid, resdir, nb_eval, dim=registered_envs[env_name]['dim_grid'])
    grid_management.dump_grid(grid, resdir, dim=registered_envs[env_name]['dim_grid'])

    return population, None, paretofront, grid



env = gym.make(registered_envs[env_name]['gym_name'], **registered_envs[env_name]['env_params'])


if (__name__ == "__main__"):

    resdir="res_"+env_name+"_"+datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    os.mkdir(resdir)
    ngen=registered_envs[env_name]['nb_gen']
    lambda_=registered_envs[env_name]['lambda']
    mu=registered_envs[env_name]['mu']

    with open(resdir+"/run_params.log", "w") as rf:
        rf.write("env_name: "+env_name)
        for k in registered_envs[env_name].keys():
            rf.write(k+": "+str(registered_envs[env_name][k])+"\n")

    pop, logbook, paretofront, grid = launch_ea(mu=mu, lambda_=lambda_, ngen=100, resdir=resdir)


    cdir="completed_runs"

    try:
        os.mkdir(cdir)
    except FileExistsError:
        pass
    
    os.rename(resdir,cdir+"/"+resdir) 

    env.close()

    print("Results saved in "+cdir+"/"+resdir)

