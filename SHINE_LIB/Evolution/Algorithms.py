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

from .Evaluator import *



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





def Simple_NSGA(nn_class,selector_class, mu=100, lambda_=200, cxpb=0.3, mutpb=0.7, ngen=100, verbose=False, resdir="res",weights=(1.0,1.0), **kwargs):
    
    random.seed()
    selector = selector_class(**kwargs)

    creator.create("MyFitness", base.Fitness, weights=weights)

    creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None)
    creator.create("Strategy", array.array, typecode="d")
    nn=nn_class(**kwargs)
    center=nn.get_parameters()
    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy, nn.get_size(), 
                     kwargs["min_value"], 
                     kwargs["max_value"], 
                     kwargs["min_strategy"], 
                     kwargs["max_strategy"])

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
    toolbox.register("select", selector.select)
    
    toolbox.register("map",futures.map)
    toolbox.decorate("mate", checkStrategy(kwargs["min_strategy"]))
    toolbox.decorate("mutate", checkStrategy(kwargs["min_strategy"]))


    toolbox.register("evaluate", Simple_Gene_Evaluator, resdir=resdir, nn_class=nn_class, **kwargs)


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
        ind.fit = fit
        ind.log = log
        ind.bd = bd
        fbd.write(" ".join(map(str,bd))+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(fit)+"\n")
        ffit.flush()

    if paretofront is not None:
        paretofront.update(population)
    
    #Améliorer ça 

    selector.update_with_offspring(population)
    selector.compute_objectifs(population)


    #print("Fit=%f Nov=%f"%(ind.fit, ind.novelty))

    indexmax, valuemax = max(enumerate([i.log[kwargs['watch_max']] for i in population]), key=operator.itemgetter(1))


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
            ind.fit = fit
            ind.bd = bd
            ind.log=log
            fbd.write(" ".join(map(str,bd))+"\n")
            fbd.flush()
            finfo.write(str(log)+"\n")
            finfo.flush()
            ffit.write(str(fit)+"\n")
            ffit.flush()

        pq=population+offspring

        selector.update_with_offspring(pq)
        selector.compute_objectifs(pq)

        #print("Fitness values: "+str(ind.fitness.values))
        

        # Select the next generation population
        population[:] = toolbox.select(pq, mu)

        indexmax, newvaluemax = max(enumerate([i.log[kwargs['watch_max']] for i in pq]), key=operator.itemgetter(1))
        if (newvaluemax>valuemax):
            valuemax=newvaluemax
            print("Gen "+str(gen)+", new max ! max fit="+str(valuemax)+" index="+str(indexmax)+" BD="+str(pq[indexmax].bd))
            nnfit, nnbd, log = Simple_Gene_Evaluator(pq[indexmax],nn_class,resdir = resdir,render=True,dump=True,name="gen%04d"%(gen),**kwargs)
    fbd.close()
    finfo.close()
    ffit.close()

    return population, selector