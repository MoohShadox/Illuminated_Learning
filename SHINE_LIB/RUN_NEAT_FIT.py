from Evolution.Algorithms import *
from Genomes.Simple_Neuron import *
from Evolution.Selector_Neuro_Evol import *
import random
from Experimentation.Experience import *
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


random.seed(20)

creator.create("MyFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None)
creator.create("Strategy", array.array, typecode="d")


def test_NEAT_FIT(resdir, **kw):
    pop,select, max_val = Neuro_Evolution(resdir, Selector_NEAT_FIT, **kw)
    select.save_stats(resdir)
    return pop, select, max_val



class NEAT_FIT(Experience):
    
    def __init__(self):
        super().__init__()

    def run(self):
        test_NEAT_FIT(resdir = self.make_resdir(algorithm=type(self).__name__), **self.kw)
        


if (__name__ == "__main__"):
    E = NEAT_FIT()
    E.run()
