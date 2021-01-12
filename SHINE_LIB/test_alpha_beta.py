from Evolution.Algorithms import *
from Genomes.Simple_Neuron import *
from Evolution.Selector import *
import random

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


random.seed(20)

creator.create("MyFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None)
creator.create("Strategy", array.array, typecode="d")


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
    'selection' : "blablabla",
    "alpha":7,
    "beta":80,
}


def test_SHINE(alpha,beta,ngen=200):
    kw["alpha"] = alpha
    kw["beta"] = beta
    resdir="res_5_10/RUN_SHINE_"+str(kw["alpha"]) + "_" + str(kw["beta"]) +"_" +datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    os.mkdir(resdir)
    pop,select, max_val = Simple_NSGA(Regression, Selector_SHINE, ngen=ngen, resdir=resdir, weights=(1.0,), **kw)
    select.save_stats(resdir)
    return pop,select, max_val


def test_SHINEDISCT(ngen=200):
    resdir="res/RUN_SHINE_DISC_"+str(kw["alpha"]) + "_" + str(kw["beta"]) +"_" +datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    os.mkdir(resdir)
    pop,select, max_val = Simple_NSGA(Regression, Selector_SHINE_DISC, ngen=ngen, resdir=resdir, weights=(1.0,), **kw)
    return pop,select, max_val

def test_NS():
    resdir="res/RUN_NS_"+"_"+datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    os.mkdir(resdir)
    pop,select = Simple_NSGA(Regression, Selector_SHINE, ngen=200, resdir=resdir, weights=(1.0,), **kw)

def test_FIT():
    resdir="res/RUN_FIT"+"_"+datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    os.mkdir(resdir)
    pop,select = Simple_NSGA(Regression, Selector_FIT, ngen=200, resdir=resdir, weights=(1.0,), **kw)

def test_MAPELITES(dim_grid):
    kw["dim_grid"] = dim_grid
    resdir="res/RUN_MAPELITES_"+str(kw["dim_grid"])+"_"+datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    os.mkdir(resdir)
    pop,select, max_val = Simple_NSGA(Regression, Selector_MAPElites, ngen=100, resdir=resdir, weights=(1.0,), **kw)
    select.save_stats(resdir)

    return max_val

def test_alpha_beta():
    perf_dict = {}
    f = open("res_5_10/output_dict_2.txt","w")
    for alpha in [9,7]:
        for beta in [90,50,30,10]:
            st = []
            for i in range(10):
                print("Experimentation for : ",alpha," and ", beta)
                print("=============")
                pop, select, max_val = test_SHINE(alpha,beta,ngen=50)
                st.append(max_val)
            st = np.array(st)
            f.write(str(alpha) + ", "+ str(beta) + " , " + " : " + str(st.mean()) + " , " + str(st.std()) + "\n" )
            perf_dict[(alpha,beta)] = (st.mean(), st.std())
    print(perf_dict)





def test_grid_MAP():
    perf_dict = {}
    f = open("grid_test.txt","w")
    for i in range(10,40,5):
        L = []
        for k in range(3):
            v = test_MAPELITES(([i,i]))
            L.append(v)
        L = np.array(L)
        f.write(str(i) + ", "+ str(i) + " , " + " : " + str(L.mean()) + " , " + str(L.std()) + "\n" )
        perf_dict[(i,i)] = (L.mean(), L.std())
    print(perf_dict)



if (__name__ == "__main__"):
    #test_SHINEDISCT(ngen=100)
    #test_grid_MAP()
    test_alpha_beta()
