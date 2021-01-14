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


def test_MAPELITES(resdir, **kw):
    pop,select, max_val = Simple_NSGA(Regression, Selector_MAPElites_POL, ngen=100, resdir=resdir, weights=(1.0,), **kw)
    select.save_stats(resdir)
    return pop, select, max_val

from Experimentation.Experience import *


class MAPElites_POL(Experience):
    
    def __init__(self):
        super().__init__()

    def run(self):
        for i in range(10):
            print("=========================")
            resdir = self.make_resdir()
            pop, select, max_val = test_MAPELITES(resdir, **self.kw)
            print("=========================")


if (__name__ == "__main__"):
    E = MAPElites_POL()
    E.run()
