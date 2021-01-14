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


def test_MAPELITES(dim_grid,resdir, **kw):
    kw["dim_grid"] = dim_grid
    pop,select, max_val = Simple_NSGA(Regression, Selector_MAPElites_COL, ngen=100, resdir=resdir, weights=(1.0,), **kw)
    select.save_stats(resdir)
    return pop, select, max_val

from Experimentation.Experience import *


class MAPElites_Experience(Experience):
    
    def __init__(self):
        super().__init__()

    def run(self):
        perf_dict = {}
        f = open(self.dir + "/summary_experience.txt","w")
        for dim in range(5,150,10):
            st = []
            for i in range(1):
                print("Experimentation for : ",dim)
                print("=========================")
                resdir = self.make_resdir(dim = dim)
                pop, select, max_val = test_MAPELITES(2*[dim],resdir, **self.kw)
                st.append(max_val)
                print("=========================")
            st = np.array(st)
            f.write( str(dim) +" : " + str(st.mean()) + " , " + str(st.std()) + "\n" )
            perf_dict[(dim,)] = (st.mean(), st.std())
        print(perf_dict)


if (__name__ == "__main__"):
    E = MAPElites_Experience()
    E.run()
