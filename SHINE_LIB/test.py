from Evolution.Algorithms import *
from Evolution.Algorithms import *
from Genomes.Simple_Neuron import *
from Evolution.Selector import *
import random

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

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
    'dim_grid': [50, 50],
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
    'selection' : "blablabla",
    "alpha":7,
    "beta":80,
}



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




if __name__=="__main__":
    resdir="res/RUN_NEURO_MAP_"+str(kw["dim_grid"])+"_"+datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    os.mkdir(resdir)
    Neuro_Evolution_MAPELITES(resdir,params,**kw)