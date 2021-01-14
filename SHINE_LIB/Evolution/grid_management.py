import random
import functools
import math
import numpy as np

def stat_grid(grid, resdir, nb_eval, dim=[100, 100]):
    if(len(grid.values())==0):
        print("Empty grid: no stats...")
        return
    nb_filled=0
    max_v=None
    for i in range(dim[0]):
        for j in range(dim[1]):
            if ((i,j) in grid.keys()):
                nb_filled+=1
                
    nbcells=functools.reduce(lambda x, y: x*y, dim)
    c_values=[ind.fit for ind in list(grid.values())]
    max_v=max(c_values)
    total_quality=sum(c_values)
    #print("Number of evaluations: %d"%(nb_eval))
    print("Coverage: %.2f %% (%d cells out of %d)"%(float(nb_filled)/float(nbcells)*100., nb_filled, nbcells)+" Max score: %.2f"%(max(c_values))+" Min score: %.2f"%(min(c_values))+" Total quality: %.2f"%(total_quality))
    stat_grid={
        'nb_eval': nb_eval,
        'nb_cells': nbcells,
        'nb_filled': nb_filled,
        'max_score': max(c_values),
        'min_score': min(c_values),
        'coverage': float(nb_filled)/float(nbcells),
        }
    with open(resdir+"/stat_grid.log","a") as sf:
        sf.write(str(stat_grid)+"\n")
    return stat_grid
        


def dump_grid(grid, resdir, dim=[100, 100]):
    if(len(grid.values())==0):
        print("Empty grid: no dump...")
        return
    with open(resdir+"/map.dat","w") as mf:
        for i in range(dim[0]):
            for j in range(dim[1]):
                if ((i,j) in grid.keys()):
                    mf.write("%.2f "%(grid[(i,j)].fit))
                else:
                    mf.write("=== ")
            mf.write("\n")
    with open(resdir+"/map_bd.dat","w") as mf:
        for p in grid.keys():
            ind=grid[p]
            for i in range(len(ind.bd)):
                mf.write("%f "%(ind.bd[i]))
            mf.write("\n")


class Grid():
    def __init__(self, mins, maxs, dims, comparator = None):
        self.min_v = mins
        self.max_v = maxs
        self.dim = dims
        self.x1 = np.linspace(self.min_v[0], self.max_v[0], self.dim[0], endpoint=False)
        self.x2 = np.linspace(self.min_v[1], self.max_v[1], self.dim[1], endpoint=False)
        self.ind_comparator = lambda ind1, ind2 : ind1.log["collision"] < ind2.log["collision"]
        if(comparator):
            self.ind_comparator = comparator
        self.content = {}
        self.stats = {}
    
    def get_grid_coord(self, ind):
        return ((self.x1 > ind.bd[0]).astype("int").argmax(),(self.x2 > ind.bd[1]).astype("int").argmax() )

    def get_from_bd(self, bd):
        return ((self.x1 > bd[0]).astype("int").argmax(),(self.x2 > bd[1]).astype("int").argmax() )

    
    def add(self,ind ):
        (x,y) = self.get_grid_coord(ind)
        if((x,y) in self.content and self.ind_comparator(ind,self.content[(x,y)]) ):
            #print((x,y) , "-> updated to : ",ind.log["collision"], " cuz it's less than : ",self.content[(x,y)].log["collision"])
            self.content[(x,y)] = ind
        elif ((x,y) not in self.content):
            self.content[(x,y)] = ind

    
    def get_stats(self, resdir, nb_eval):
        self.stats = stat_grid(self.content, resdir, nb_eval, self.dim)
        return self.stats
        

    def dump(self, resdir):
        dump_grid(self.content, resdir, self.dim)
        pass
        
def cart2pol(x, y, goal = [60,60]):
    rho = np.sqrt((x-goal[0])**2 + (y-goal[1])**2)
    phi = np.arctan2((y-goal[1])**2, (x-goal[0])**2)
    return(np.log(rho), phi)

def get_pol_coord(x,y):
    deg = np.linspace(0, 3.14, 150, endpoint=False)
    dist = np.linspace(0, np.log(600*600), 150, endpoint=False)
    rho, phi = cart2pol(x,y)
    return (rho < dist).argmax(), (deg > phi).argmax()

    

class Grid_POL():

    def __init__(self, mins, maxs, dims, comparator = None):
        self.min_v = mins
        self.max_v = maxs
        self.dim = dims
        self.ind_comparator = lambda ind1, ind2 : ind1.log["collision"] < ind2.log["collision"]
        if(comparator):
            self.ind_comparator = comparator
        self.content = {}
        self.stats = {}

    
    def get_grid_coord(self, ind):
        deg = np.linspace(0, 3.14, 150, endpoint=False)
        dist = np.linspace(-2, np.log(600*600), 150, endpoint=False)
        dist = np.exp(dist)
        rho, phi = cart2pol(ind.bd[0],ind.bd[1])
        return (rho < dist).argmax(), (deg > phi).argmax()


    
    def add(self,ind ):
        (x,y) = self.get_grid_coord(ind)
        if((x,y) in self.content and self.ind_comparator(ind,self.content[(x,y)]) ):
            #print((x,y) , "-> updated to : ",ind.log["collision"], " cuz it's less than : ",self.content[(x,y)].log["collision"])
            self.content[(x,y)] = ind
        elif ((x,y) not in self.content):
            self.content[(x,y)] = ind

    
    def get_stats(self, resdir, nb_eval):
        self.stats = stat_grid(self.content, resdir, nb_eval, self.dim)
        return self.stats
        

    def dump(self, resdir):
        dump_grid(self.content, resdir, self.dim)
        pass
        






