from .grid_management import Grid
import numpy as np
from .quadtree import *
from .Behaviour_Descriptor import Behaviour_Descriptor
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from scipy.spatial import KDTree

import deap
import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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

#from novelty_search_vanila import *
import os


class Novelty_Archive():
    def __init__(self, k=15, lambda_=10, **kwargs):
        self.all_bd={}
        self.kdtree=None
        self.k=k
        self.lambda_ = lambda_
        #print("Archive constructor. size = %d"%(len(self.all_bd)))

    def update(self,ind):
        oldsize=len(self.all_bd)
        if(self.kdtree == None):
            self.all_bd[tuple(ind.bd)] = self.all_bd.get(tuple(ind.bd),[]) + [ind]
            self.kdtree=KDTree([ind.bd])
        else:
            self.all_bd[tuple(ind.bd)] = self.all_bd.get(tuple(ind.bd),[]) + [ind]
            self.kdtree=KDTree(list(self.all_bd.keys()))
        #print("Archive updated, old size = %d, new size = %d"%(oldsize,len(self.all_bd)))
    
    
    def select_from_offsprint(self, offspring):
        soff=sorted(offspring,key=lambda x:x.novelty)
        ilast=len(offspring)-self.lambda_
        lbd=[soff[i] for i in range(ilast,len(soff))]
        return lbd
    
    def update_offspring(self, offspring):
        if(self.kdtree == None):
            self.update(offspring[0])
        self.apply_novelty_estimation(offspring)
        lbd = self.select_from_offsprint(offspring)
        for i in lbd:
            self.update(i)

    def apply_novelty_estimation(self, population):
        for ind in population:
            ind.novelty = self.get_nov(ind.bd, population)
        

    def get_nov(self,bd, population=[]):
        dpop=[]
        for ind in population:
            dpop.append(np.linalg.norm(np.array(bd)-np.array(ind.bd)))
        darch,ind=self.kdtree.query(np.array(bd),self.k)
        d=dpop+list(darch)
        d.sort()
        if (d[0]!=0):
            print("WARNING in novelty search: the smallest distance should be 0 (distance to itself).")
        return sum(d[:self.k+1])/self.k # as the indiv is in the population, the first value is necessarily a 0.

    def size(self):
        return len(self.all_bd)



class Novelty_Archive_random(Novelty_Archive):

    def __init__(self, k=15, lambda_ = 10):
        super().__init__(k=k, lambda_=lambda_)
    
    def select_from_offsprint(self, offspring, verbose=False):
        l=list(range(len(offspring)))
        random.shuffle(l)
        if (verbose):
            print("Random archive update. Adding offspring: "+str(l[:lambda_])) 
        lbd=[offspring[l[i]] for i in range(self.lambda_)]
        return lbd


class Shine_Archive(QuadTree):
    def __init__(self, width, height, alpha = 7, beta = 10):
        super().__init__([],width, height)
        rect = Rect(0, 0, width, height)
        self.alpha = alpha
        self.beta = beta
        self.size = 0
        self.root = Node(val=[], bounds=rect)

    
    def update_offspring(self,offspring):
        for i in offspring:
            I = Behaviour_Descriptor(i)
            self.add_node(I)
        

    def _split(self, root):
        if root.leaf:
            rects = root.bounds.split()
            for son, bounds_rect in zip(("nw", "ne", "sw", "se"), rects):
                setattr(root, son, Node(val=[], bounds=bounds_rect, level=root.level + 1))
            for val in root.val:
                for son in root.sons():
                    if val in root.bounds:
                        son.val.append(val)
                        if(len(son.val) > self.beta):
                            self._split(son)
                        break
            root.val.clear()
        else:
            print("Trying to split not a leaf")
    
    def remove_worst(self, node):
        if(len(node.val) == 0):
            return
        points = [(i[0], i[1]) for i in node.val]
        dists = [ (i,np.sqrt((i[0] - (node.bounds[0] + node.bounds[2]))**2 +  (i[1] - (node.bounds[1] + node.bounds[3])) **2 )) for i in node.val]
        #dists = [ (i,i.ind.fit) for i in node.val]
        #print("points : ", points)
        #print(dists)
        dists = (sorted(dists, key = lambda x:x[1]))
        pos = node.val.index(dists[0][0])
        node.val.remove(dists[0][0])
        #print("points after : ",node.val)
        pass


    def add_node(self, val):
        node = self.search(val)
        if(val not in node.val):
            node.val.append(val)
        if(len(node.val) >= self.beta):
            if(node.level <= self.alpha):
                self._split(node)
                self.size += 1
            else:
                self.remove_worst(node)
        
        if(len(node.val) > self.beta):
            print("size after : ",len(node.val), " , level : ",node.level)

