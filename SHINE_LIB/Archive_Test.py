from Evolution.Archive import *
from Evolution.Behaviour_Descriptor import *

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


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

import numpy as np

if (__name__ == "__main__"):
    random.seed()
    creator.create("MyFitness", base.Fitness, weights=(1.0,1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None)
    creator.create("Strategy", array.array, typecode="d")
    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,10, 
                     -1, 
                     1, 
                     -1, 
                     1)
    Sh = Shine_Archive(1000,1000)
    ints = np.random.randint(0,1000,(1000,2))
    L = []
    for i in ints:
        I = toolbox.individual()
        I.bd = i
        L.append(I)
    Sh.update_offspring(L)
    print("Verification du respect de alpha et de beta ")
    for i in Sh:
        if(i.level > Sh.alpha):
            print('Error level = ',level," and alpha = ", Sh.alpha)
        if(len(i.val) > Sh.beta):
            print('Error level = ',len(i.val)," and alpha = ", Sh.beta)
    print("Checked", u"\u2713")
    print("Verification du niveau de chaque noeud")
    for i in Sh:
        for son in i.sons():
            if(son):
                assert (i.level +1== son.level )
    print("Checked", u"\u2713")

