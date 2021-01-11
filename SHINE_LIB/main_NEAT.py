from Evolution.Algorithms import *
from Genomes.Simple_Neuron import *
from Evolution.Selector import *
import random

import math
import time

import MultiNEAT as NEAT
import gym, gym_fastsim

import progressbar as pbar
import numpy as np



kw={
    'gym_name': 'FastsimSimpleNavigation-v0',
    'env_params': {"still_limit": 10, 'reward_kind': "continuous"},

    'nb_input': 5, # number of NN inputs
    'nb_output': 2, # number of NN outputs
    'nb_layers': 2, # number of layers
    'nb_neurons_per_layer': 10, # number of neurons per layer
    'max_action': 2,
    'episode_nb_step': 1000, 
    'episode_reward_kind': 'final', 
    'episode_bd': 'robot_pos', 
    'episode_bd_slice': (0,2,None), 
    'episode_bd_kind': 'final', 
    'episode_log': {'collision': 'cumul', 
                    'dist_obj': 'final', 
                    'exit_reached': 'final', 
                    'robot_pos': 'final'},
    'dim_grid': [100, 100],
    'grid_min_v': [0,0],
    'grid_max_v': [600,600],
    'goal': [60,60],
    'watch_max': 'dist_obj',
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
    'selection' : "blablabla"
}



class Behavior:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return np.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)


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


max_evaluations = 30000

screen_size_x, screen_size_y = 600, 600
max_timesteps = 2000



params = NEAT.Parameters()
params.PopulationSize = 100
params.DynamicCompatibility = True
params.AllowClones = False
params.AllowLoops = False
params.CompatTreshold = 5.0
params.CompatTresholdModifier = 0.3
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 100
params.OldAgeTreshold = 35
params.MinSpecies = 3
params.MaxSpecies = 10
params.RouletteWheelSelection = True
params.RecurrentProb = 0.0
params.OverallMutationRate = 0.02
params.MutateWeightsProb = 0.90
params.WeightMutationMaxPower = 1.0
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.5
params.WeightMutationRate = 0.75
params.MaxWeight = 8
params.MutateAddNeuronProb = 0.01
params.MutateAddLinkProb = 0.02
params.MutateRemLinkProb = 0.00

params.Elitism = 0.1
params.CrossoverRate = 0.5
params.MutateWeightsSevereProb = 0.01

params.MutateNeuronTraitsProb = 0
params.MutateLinkTraitsProb = 0

rng = NEAT.RNG()
rng.TimeSeed()

env = gym.make('FastsimSimpleNavigation-v0')
env.reset()




import random as rnd
import sys
import pickle


def Test_Neat():

    g = NEAT.Genome(0, 5, 2, 2, False,
                    NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params, 0)
    
    pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 1000))

    print('NumLinks:', g.NumLinks())
    #sys.exit(0)

    best_genome_ever = None
    best_ever = -np.inf
    fast_mode = True
    evhist = []
    best_gs = []
    hof = []

    if not ns_on:

        # rtNEAT mode
        print('============================================================')
        print("Please wait for the initial evaluation to complete.")
        fitnesses = []
        
        for _, genome in enumerate(NEAT.GetGenomeList(pop)):
            fitness,behaviour,logs = Neuro_Evolve_Evaluation(genome, **kw)
            print('Evaluating',_, " : ",fitness)
            fitnesses.append(fitness)
        for genome, fitness in zip(NEAT.GetGenomeList(pop), fitnesses):
            genome.SetFitness(fitness)
            genome.SetEvaluated()
        maxf = max([x.GetFitness() for x in NEAT.GetGenomeList(pop)])
        print("Max f au debut : ", maxf)
        print('======================')
        print('rtNEAT phase')
        pb = pbar.ProgressBar(maxval=max_evaluations)
        pb.start()
        for i in range(max_evaluations):
            # get best fitness in population and print it
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

            # get the new baby
            old = NEAT.Genome()
            baby = pop.Tick(old)

            # evaluate it
            f,bh,_ = Neuro_Evolve_Evaluation(baby,**kw)
            #print("fitness baby : ", f)
            baby.SetFitness(f)
            baby.SetEvaluated()
            fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
            #print("Fitness list at end of step : ", fitness_list)

            pb.update(i)
            sys.stdout.flush()
    else:

        archive = []

        # novelty search
        print('============================================================')
        print("Please wait for the initial evaluation to complete.")
        fitnesses = []
        for _, genome in enumerate(NEAT.GetGenomeList(pop)):
            print('Evaluating',_)
            fitness, behavior, _ = Neuro_Evolve_Evaluation(genome, **kw)
            # associate the behavior with the genome
            genome.behavior = behavior

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

            # compute the new behavior
            fitness, behavior, log = Neuro_Evolve_Evaluation(new, **kw)
            new.behavior = behavior

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
            if fitness > best_ever:
                sys.stdout.flush()
                print()
                print('Nouveau record  !')
                print('Evaluations:', evaluations, 'Species:', len(pop.Species), 'Fitness:',np.sqrt(kw["grid_max_v"][0]**2 + kw["grid_max_v"][0]**2)-fitness)
                hof.append(pickle.dumps(new))
                best_ever = fitness

        pb.finish()

    # Show the best genome's performance forever
    while True:
        hg = pickle.loads(hof[-1])
        evaluate(hg)



if (__name__ == "__main__"):
    Test_Neat()