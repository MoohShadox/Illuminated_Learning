from QD_Algorithms.Experience import *
from QD_Algorithms.Genotype import *
from Archive import *
import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from QD_Algorithms.fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
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
from scoop import futures
import os



def simple_evaluation_generator(criterion,max_steps, epochs, genotype_model):
    G = genotype_model()
    exp1 = Experience()
    def eval_f(params):
        G = genotype_model()
        G.set_params(params)
        eval_ = exp1.simple_evaluation(G, criterion=criterion,max_steps=max_steps,epochs=epochs)[0]
        return eval_
    return eval_f,G.get_params().shape,exp1


def mo_evaluation_generator(criterions,max_steps, epochs, genotype_model):
    G = genotype_model()
    exp1 = Experience()
    def eval_f(params):
        G = genotype_model()
        G.set_params(params)
        eval_, runs = exp1.mo_evaluation(G, criterions=criterions,max_steps=max_steps,epochs=100)
        return tuple(i for i,j in eval_.values()), eval_, runs, G
    return eval_f,G.get_params().shape,exp1


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








class Evolutionnary_Algorithm(object):

    def __init__(self, genotype_model,phenotypic_descriptor=None, *args):
        super(Evolutionnary_Algorithm, self).__init__(*args)
        self.genotype_model = genotype_model
        self.experience = None
        self.eval_function = None
        self.population = None
        self.toolbox = None
        self.objectifs = None
        self.spec = {}
        self.nb_eval = 0
        self.paretofront = None
        self.read("conf/conf.yaml")
        self.objectifs = self.spec["objectifs"]
        self.ponderations = self.spec["ponderations"]
        self.phenotypic_descriptor = phenotypic_descriptor
        if(self.phenotypic_descriptor):
            self.archive = Archive(self.phenotypic_descriptor)
    
    

    def read(self,file_path : str):
        """Construct an evaluator from a YAML file.
        Args:
            file_path (str): path to the yaml file.
        """
        self.env = {}
        with open(file_path) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            self.spec.update(list(docs)[0]["ea_spec"])
    

    def write(self,file_path : str):
        with open(file_path,"w") as f:
            docs = yaml.dump(self.env,f)

    def initialise(self,path=None):
        random.seed()
        creator.create("MyFitness", base.Fitness, weights=self.spec["ponderations"])
        creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None)
        creator.create("Strategy", array.array, typecode="d")

        func, ind_size, self.experience = mo_evaluation_generator(self.spec["objectifs"],self.spec["max_steps"], self.spec["epochs"], self.genotype_model)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", generateES, creator.Individual, creator.Strategy, ind_size[0], 
                         self.spec["min_value"], 
                         self.spec["max_value"], 
                         self.spec["min_strategy"], 
                         self.spec["max_strategy"])
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxESBlend, alpha=self.spec["alpha"])
        self.toolbox.register("mutate", tools.mutESLogNormal, c=self.spec["c"], indpb=self.spec["indpb"])
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("map",futures.map)
        self.toolbox.decorate("mate", checkStrategy(self.spec["min_strategy"]))
        self.toolbox.decorate("mutate", checkStrategy(self.spec["min_strategy"]))
        self.toolbox.register("evaluate", func)
        self.paretofront = tools.ParetoFront()

    def create_population(self):
        self.population = self.toolbox.population(n=self.spec["mu"])
        self.nb_eval=0

    def evaluate(self,individuals):
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        fitnesses_bds = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        self.nb_eval+=len(invalid_ind)

        for ind, (fit, log, runs, G) in zip(invalid_ind, fitnesses_bds):
            ind.fitness.values=fit
            ind.fit = fit
            ind.log = runs
            ind.genotype = G

        if(self.phenotypic_descriptor):
            self.archive.novelty_reset()
            for ind in individuals:
                self.archive.add(ind.genotype)
            self.archive.compute_all_novelty()
            for ind in individuals:
                novelty = self.archive.get_novelty(ind.genotype)
                ind.novelty = novelty
                ind.fitness.values = ind.fitness.values + novelty

        
    
    def evolve(self,verbose=False):
        self.evaluate(self.population)
        offspring = algorithms.varOr(self.population, self.toolbox, self.spec["lambda"], self.spec["cxpb"], self.spec["mutpb"])
        pq=self.population+offspring
        self.evaluate(pq)
        self.population[:] = self.toolbox.select(pq, self.spec["mu"])
        if self.paretofront is not None:
            self.paretofront.update(self.population)
        
        valuemax = -np.inf
        indexmax, newvaluemax = max(enumerate([i.log[self.experience.env['watch_max']][0] for i in pq]), key=operator.itemgetter(1))
        print("new value max : ",newvaluemax)
        if (newvaluemax>valuemax):
            valuemax=newvaluemax
            print(" new max ! max fit="+str(valuemax)+" index="+str(indexmax))
    
    def run(self):
        for gen in range(self.spec["nb_gen"]):
            self.evolve()


if(__name__=="__main__"):
    ea = Evolutionnary_Algorithm(Simple_Genotype)
    ea.initialise()
    ea.create_population()
    ea.run()
