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





def test_SHINEDISCT(resdir, ngen=200, **kw):
    pop,select, max_val = Simple_NSGA(Regression, Selector_SHINE_DISC, ngen=ngen, resdir=resdir, weights=(-1.0,-1.0), **kw)
    return pop,select, max_val


from Experimentation.Experience import *



class SHINE_MO(Experience):

    def __init__(self):
        super().__init__()

    def run(self):
        perf_dict = {}
        f = open(self.dir + "/summary_experience.txt","w")
        st = []
        alpha = 7
        beta = 30
        for i in range(20):
            print("Experimentation for : ",alpha," and ", beta)
            print("=============")
            pop, select, max_val = test_SHINEDISCT( self.make_resdir(alpha=alpha, beta=beta) ,ngen=200, **self.kw)
            st.append(max_val)
        st = np.array(st)
        f.write(str(alpha) + ", "+ str(beta) + " , " + " : " + str(st.mean()) + " , " + str(st.std()) + "\n" )
        perf_dict[(alpha,beta)] = (st.mean(), st.std())
        print(perf_dict)



if (__name__ == "__main__"):
    E = SHINE_MO()
    E.run()
