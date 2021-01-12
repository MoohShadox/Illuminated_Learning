import deap
import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy


import random as rnd
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from Evolution.Algorithms import *
from Genomes.Simple_Neuron import *
from Evolution.Selector import *
import random
from Evolution.grid_management import Grid
import math
import time

import MultiNEAT as NEAT
import gym, gym_fastsim

import progressbar as pbar
import numpy as np

import datetime
import random as rnd

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

    return population, selector, valuemax


def Neuro_Evolution_FIT(resdir,params,max_evaluations = 30000,**kw):
    g = NEAT.Genome(0, 5, 2, 2, False,NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params, 0)
    
    rng = NEAT.RNG()
    rng.TimeSeed()
    pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 1000))

    print('NumLinks:', g.NumLinks())
    #sys.exit(0)
    fbd=open(resdir+"/bd.log","w")
    finfo=open(resdir+"/info.log","w")
    ffit=open(resdir+"/fit.log","w")
    best_genome_ever = None
    best_ever = -np.inf
    fast_mode = True
    evhist = []
    best_gs = []
    hof = []


    # rtNEAT mode
    print('============================================================')
    print("Please wait for the initial evaluation to complete.")
    fitnesses = []
    finfo.write("## Generation 0 \n")
    finfo.flush()
    ffit.write("## Generation 0 \n")
    ffit.flush()
    for _, genome in enumerate(NEAT.GetGenomeList(pop)):
        fit,bd,log = Neuro_Evolve_Evaluation(genome,resdir=resdir, **kw)
        print('Evaluating',_, " : ",fit)
        fbd.write(str(bd[0])+" "+str(bd[1])+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(fit)+"\n")
        ffit.flush()
        fitnesses.append(fit)

    for genome, fitness in zip(NEAT.GetGenomeList(pop), fitnesses):
        genome.SetFitness(fitness)
        genome.SetEvaluated()


    maxf = max([x.GetFitness() for x in NEAT.GetGenomeList(pop)])
    pb = pbar.ProgressBar(maxval=max_evaluations)
    pb.start()

    for i in range(max_evaluations):

        fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
        best = max(fitness_list)
        #print("Fitness list : ", fitness_list)
        #print("Max  : ", best)
        evhist.append(best)
        
        if best > best_ever:
            sys.stdout.flush()
            print()
            print('NEW RECORD!')
            print('Evaluations:', i, 'Species:', len(pop.Species), 'Fitness:',np.sqrt(kw["grid_max_v"][0]**2 + kw["grid_max_v"][0]**2)-best)
            best_gs.append(pop.GetBestGenome())
            best_ever = best
            hof.append(pickle.dumps(pop.GetBestGenome()))
            Neuro_Evolve_Evaluation(pop.GetBestGenome(),resdir=resdir,name="gen%04d"%(i), dump=True, **kw)
        

        # get the new baby
        old = NEAT.Genome()
        baby = pop.Tick(old)
        # evaluate it
        f,bd,log = Neuro_Evolve_Evaluation(baby,resdir=resdir,**kw)

        finfo.write("## Generation %d \n"%(i))
        finfo.flush()
        ffit.write("## Generation %d \n"%(i))
        ffit.flush()
        
        fbd.write(str(bd[0])+" "+str(bd[1])+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(f)+"\n")
        ffit.flush()


        baby.SetFitness(f)
        baby.SetEvaluated()
        fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
        pb.update(i)
        sys.stdout.flush()


ns_on = 1

ns_K = 30
ns_recompute_sparseness_each = 20
ns_P_min = 10.0
ns_dynamic_Pmin = True
ns_Pmin_min = 1.0
ns_no_archiving_stagnation_threshold = 150
ns_Pmin_lowering_multiplier = 0.9
ns_Pmin_raising_multiplier = 1.1
ns_quick_archiving_min_evals = 8


def Neuro_Evolution_NS(resdir,params,max_evaluations = 30000,**kw):
    g = NEAT.Genome(0, 5, 2, 2, False,NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params, 0)
    
    rng = NEAT.RNG()
    rng.TimeSeed()
    pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 1000))

    print('NumLinks:', g.NumLinks())
    #sys.exit(0)
    fbd=open(resdir+"/bd.log","w")
    finfo=open(resdir+"/info.log","w")
    ffit=open(resdir+"/fit.log","w")
    best_genome_ever = None
    best_ever = -np.inf
    fast_mode = True
    evhist = []
    best_gs = []
    hof = []


    # rtNEAT mode
    print('============================================================')
    print("Please wait for the initial evaluation to complete.")
    fitnesses = []
    finfo.write("## Generation 0 \n")
    finfo.flush()
    ffit.write("## Generation 0 \n")
    ffit.flush()

    archive = []
    # novelty search
    print('============================================================')
    print("Please wait for the initial evaluation to complete.")
    fitnesses = []
    for _, genome in enumerate(NEAT.GetGenomeList(pop)):
        print('Evaluating',_)
        fit,bd,log = Neuro_Evolve_Evaluation(genome,resdir=resdir, **kw)
        print('Evaluating',_, " : ",fit)
        fbd.write(str(bd[0])+" "+str(bd[1])+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(fit)+"\n")
        ffit.flush()
        fitnesses.append(fit)
        # associate the behavior with the genome
        genome.behavior = bd
    # recompute sparseness
    def sparseness(genome):
        distances = []
        for g in NEAT.GetGenomeList(pop):
            d = genome.behavior.distance_to( g.behavior )
            distances.append(d)
        # get the distances from the archive as well
        for ab in archive:
            distances.append( genome.behavior.distance_to(ab) )
        distances = sorted(distances)
        sp = np.mean(distances[1:ns_K+1])
        return sp
    print('======================')
    print('Novelty Search phase')
    pb = pbar.ProgressBar(maxval=max_evaluations)
    # Novelty Search variables
    evaluations = 0
    evals_since_last_archiving = 0
    quick_add_counter = 0
    # initial fitness assignment
    for _, genome in enumerate(NEAT.GetGenomeList(pop)):
        genome.SetFitness( sparseness(genome) )
        genome.SetEvaluated()
    # the Novelty Search tick
    while evaluations < max_evaluations:
        pb.start()
        global ns_P_min
        evaluations += 1
        pb.update(evaluations)
        sys.stdout.flush()
        # recompute sparseness for each individual
        if evaluations % ns_recompute_sparseness_each == 0:
            for _, genome in enumerate(NEAT.GetGenomeList(pop)):
                genome.SetFitness( sparseness(genome) )
                genome.SetEvaluated()
        # tick
        old = NEAT.Genome()
        new = pop.Tick(old)

        f,bd,log = Neuro_Evolve_Evaluation(new,resdir=resdir,**kw)

        finfo.write("## Generation %d \n"%(evaluations))
        finfo.flush()
        ffit.write("## Generation %d \n"%(evaluations))
        ffit.flush()
        
        fbd.write(str(bd[0])+" "+str(bd[1])+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(f)+"\n")
        ffit.flush()
        # compute the new behavior
        new.behavior = bd
        # compute sparseness
        sp = sparseness(new)
        # add behavior to archive if above threshold
        evals_since_last_archiving += 1
        if sp > ns_P_min:
            archive.append(new.behavior)
            evals_since_last_archiving = 0
            quick_add_counter += 1
        else:
            quick_add_counter = 0
        if ns_dynamic_Pmin:
            if evals_since_last_archiving > ns_no_archiving_stagnation_threshold:
                ns_P_min *= ns_Pmin_lowering_multiplier
                if ns_P_min < ns_Pmin_min:
                    ns_P_min = ns_Pmin_min
            # too much additions one after another?
            if quick_add_counter > ns_quick_archiving_min_evals:
                ns_P_min *= ns_Pmin_raising_multiplier
        # set the fitness of the new individual
        new.SetFitness(sp)
        new.SetEvaluated()
        # still use the objective search's fitness to know which genome is best
        if f > best_ever:
            sys.stdout.flush()
            print()
            print('Nouveau record  !')
            print('Evaluations:', evaluations, 'Species:', len(pop.Species), 'Fitness:',np.sqrt(kw["grid_max_v"][0]**2 + kw["grid_max_v"][0]**2)-f)
            hof.append(pickle.dumps(new))
            Neuro_Evolve_Evaluation(new,resdir=resdir,name="gen%04d"%(evaluations), dump=True, **kw)

            best_ever = f
    pb.finish()

def Neuro_Evolution_FIT(resdir,params,max_evaluations = 30000,**kw):
    g = NEAT.Genome(0, 5, 2, 2, False,NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params, 0)
    
    rng = NEAT.RNG()
    rng.TimeSeed()
    pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 1000))

    print('NumLinks:', g.NumLinks())
    #sys.exit(0)
    fbd=open(resdir+"/bd.log","w")
    finfo=open(resdir+"/info.log","w")
    ffit=open(resdir+"/fit.log","w")
    best_genome_ever = None
    best_ever = -np.inf
    fast_mode = True
    evhist = []
    best_gs = []
    hof = []


    # rtNEAT mode
    print('============================================================')
    print("Please wait for the initial evaluation to complete.")
    fitnesses = []
    finfo.write("## Generation 0 \n")
    finfo.flush()
    ffit.write("## Generation 0 \n")
    ffit.flush()
    for _, genome in enumerate(NEAT.GetGenomeList(pop)):
        fit,bd,log = Neuro_Evolve_Evaluation(genome,resdir=resdir, **kw)
        print('Evaluating',_, " : ",fit)
        fbd.write(str(bd[0])+" "+str(bd[1])+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(fit)+"\n")
        ffit.flush()
        fitnesses.append(fit)

    for genome, fitness in zip(NEAT.GetGenomeList(pop), fitnesses):
        genome.SetFitness(fitness)
        genome.SetEvaluated()


    maxf = max([x.GetFitness() for x in NEAT.GetGenomeList(pop)])
    pb = pbar.ProgressBar(maxval=max_evaluations)
    pb.start()

    for i in range(max_evaluations):

        fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
        best = max(fitness_list)
        #print("Fitness list : ", fitness_list)
        #print("Max  : ", best)
        evhist.append(best)
        
        if best > best_ever:
            sys.stdout.flush()
            print()
            print('NEW RECORD!')
            print('Evaluations:', i, 'Species:', len(pop.Species), 'Fitness:',np.sqrt(kw["grid_max_v"][0]**2 + kw["grid_max_v"][0]**2)-best)
            best_gs.append(pop.GetBestGenome())
            best_ever = best
            hof.append(pickle.dumps(pop.GetBestGenome()))
            Neuro_Evolve_Evaluation(pop.GetBestGenome(),resdir=resdir,name="gen%04d"%(i), dump=True, **kw)
        

        # get the new baby
        old = NEAT.Genome()
        baby = pop.Tick(old)
        # evaluate it
        f,bd,log = Neuro_Evolve_Evaluation(baby,resdir=resdir,**kw)

        finfo.write("## Generation %d \n"%(i))
        finfo.flush()
        ffit.write("## Generation %d \n"%(i))
        ffit.flush()
        
        fbd.write(str(bd[0])+" "+str(bd[1])+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(f)+"\n")
        ffit.flush()


        baby.SetFitness(f)
        baby.SetEvaluated()
        fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
        pb.update(i)
        sys.stdout.flush()


def Neuro_Evolution_MAPELITES(resdir,params,max_evaluations = 30000,**kw):
    g = NEAT.Genome(0, 5, 2, 2, False,NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params, 0)
    grid=Grid(kw["grid_min_v"],kw["grid_max_v"],kw["dim_grid"],comparator = lambda ind1, ind2 : ind1.fit > ind2.fit )
    rng = NEAT.RNG()
    rng.TimeSeed()
    pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 1000))

    print('NumLinks:', g.NumLinks())
    #sys.exit(0)
    fbd=open(resdir+"/bd.log","w")
    finfo=open(resdir+"/info.log","w")
    ffit=open(resdir+"/fit.log","w")
    best_genome_ever = None
    best_ever = -np.inf
    fast_mode = True
    evhist = []
    best_gs = []
    hof = []


    # rtNEAT mode
    print('============================================================')
    print("Please wait for the initial evaluation to complete.")
    fitnesses = []
    finfo.write("## Generation 0 \n")
    finfo.flush()
    ffit.write("## Generation 0 \n")
    ffit.flush()
    for _, genome in enumerate(NEAT.GetGenomeList(pop)):
        fit,bd,log = Neuro_Evolve_Evaluation(genome,resdir=resdir, **kw)
        print('Evaluating',_, " : ",fit)


        k1,k2 = grid.get_from_bd(bd)
        #grid.content[(k1,k2)] = genome
        genome.bd = bd
        genome.fit = fit
        genome.log = log
        genome.SetFitness(fit)
        genome.SetEvaluated()
        grid.add(genome)

        fbd.write(str(bd[0])+" "+str(bd[1])+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(fit)+"\n")
        ffit.flush()
        fitnesses.append(fit)

    for genome, fitness in zip(NEAT.GetGenomeList(pop), fitnesses):
        genome.SetFitness(fitness)
        genome.SetEvaluated()


    maxf = max([x.GetFitness() for x in NEAT.GetGenomeList(pop)])
    pb = pbar.ProgressBar(maxval=max_evaluations)
    pb.start()

    for i in range(max_evaluations):

        fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
        best = max(fitness_list)
        #print("Fitness list : ", fitness_list)
        #print("Max  : ", best)
        evhist.append(best)
        
        if best > best_ever:
            sys.stdout.flush()
            print()
            print('NEW RECORD!')
            print('Evaluations:', i, 'Species:', len(pop.Species), 'Fitness:',np.sqrt(kw["grid_max_v"][0]**2 + kw["grid_max_v"][0]**2)-best)
            best_gs.append(pop.GetBestGenome())
            best_ever = best
            hof.append(pickle.dumps(pop.GetBestGenome()))
            Neuro_Evolve_Evaluation(pop.GetBestGenome(),resdir=resdir,name="gen%04d"%(i), dump=True, **kw)
        

        # get the new baby
        old = NEAT.Genome()
        baby = pop.Tick(old)
        # evaluate it
        f,bd,log = Neuro_Evolve_Evaluation(baby,resdir=resdir,**kw)

        finfo.write("## Generation %d \n"%(i))
        finfo.flush()
        ffit.write("## Generation %d \n"%(i))
        ffit.flush()
        
        fbd.write(str(bd[0])+" "+str(bd[1])+"\n")
        fbd.flush()
        finfo.write(str(log)+"\n")
        finfo.flush()
        ffit.write(str(f)+"\n")
        ffit.flush()


        baby.SetFitness(f)
        baby.SetEvaluated()

        k1,k2 = grid.get_from_bd(bd)
        grid.content[(k1,k2)] = baby
        baby.behavior = bd
        baby.SetFitness(fit)
        baby.SetEvaluated()

        pop = grid.content.values()
        pop = NEAT.Population(pop)

        fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
        pb.update(i)
        sys.stdout.flush()