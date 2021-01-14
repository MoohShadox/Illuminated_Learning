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




def test_SHINE(alpha,beta,resdir,ngen=200,**kw):
    kw["alpha"] = alpha
    kw["beta"] = beta
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

def test_MAPELITES(dim_grid,resdir):
    kw["dim_grid"] = dim_grid
    resdir="res/RUN_MAPELITES_"+str(kw["dim_grid"])+"_"+datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    os.mkdir(resdir)
    pop,select, max_val = Simple_NSGA(Regression, Selector_MAPElites, ngen=100, resdir=resdir, weights=(1.0,), **kw)
    select.save_stats(resdir)
    return pop,select, max_val

def test_NS(resdir,**kw):
    pop,select, max_val = Simple_NSGA(Regression, Selector_NS, resdir=resdir, weights=(1.0,), **kw)
    return pop,select, max_val

def test_alpha_beta():
    perf_dict = {}
    f = open("output_dict.txt","w")
    for alpha in range(3,10,2):
        for beta in range(10,100,20):
            st = []
            for i in range(2):
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



from Experimentation.Experience import *



class NS(Experience):

    def __init__(self):
        super().__init__()

    def run(self):
        perf_dict = {}
        f = open(self.dir + "/summary_experience.txt","w")
        st = []
        for i in range(10):
            print("=============")
            pop, select, max_val = test_NS(self.make_resdir(ngen=200) ,ngen=200, **self.kw)
            st.append(max_val)
        st = np.array(st)
        f.write(str(i) + " , " + " : " + str(st) + "\n" )
        perf_dict[(alpha,beta)] = st
        print(perf_dict)



if (__name__ == "__main__"):

    E = NS()
    E.run()
