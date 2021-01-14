
from .Archive import *
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from .grid_management import *


class Selector_NEAT():
    def __init__(self, archive_class, **kwargs):
        self.archive = archive_class(k=kwargs["nov_k"], lambda_ = kwargs["nov_lambda"])
        self.grid = Grid(kwargs["grid_min_v"],kwargs["grid_max_v"],kwargs["dim_grid"])

    def update(self, ind):
        self.grid.add(ind)
        self.archive.update_offspring([ind])

    def compute_objectif(self, ind):
        pass

    def update_all(self, pop):
        pass

    
    def save_stats(self,resdir):
        self.grid.dump(resdir)
        self.grid.get_stats(resdir, 1000)




class Selector_NEAT_FIT(Selector_NEAT):
    def __init__(self , **kwargs):
        self.grid = Grid(kwargs["grid_min_v"],kwargs["grid_max_v"],kwargs["dim_grid"])

    def update(self, ind):
        self.grid.add(ind)

    def compute_objectif(self, ind):
        ind.SetFitness(ind.fit)
        ind.SetEvaluated()

    
    def save_stats(self,resdir):
        self.grid.dump(resdir)
        self.grid.get_stats(resdir, 1000)


class Selector_NEAT_SHINE(Selector_NEAT):
    def __init__(self , **kwargs):
        self.grid = Grid(kwargs["grid_min_v"],kwargs["grid_max_v"],kwargs["dim_grid"])
        self.archive = Shine_Archive(600,600,alpha=kwargs["alpha"],beta=kwargs["beta"])

    def update(self, ind):
        self.grid.add(ind)
        self.archive.update_offspring([ind])


    def update_all(self, pop):
        for i in pop:
            self.compute_objectif(i)
        pass

    def compute_objectif(self, ind):
        n = self.archive.search(Behaviour_Descriptor(ind))
        if(len(n.val)> 0):
            ind.obj.SetFitness(self.archive.beta / (self.archive.beta*n.level + len(n.val) ))
        else:
            ind.obj.SetFitness(-np.inf)
        ind.obj.SetEvaluated()


    
    def save_stats(self,resdir):
        self.grid.dump(resdir)
        self.grid.get_stats(resdir, 1000)
