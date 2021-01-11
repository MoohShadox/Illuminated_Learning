import numpy as np
import pandas as pd
from QD_Algorithms.Experience import *
from QD_Algorithms.Genotype import *

class Phenotypic_Descriptor(object):
    def __init__(self, *args):
        super(Phenotypic_Descriptor, self).__init__(*args)
        self.params = {}
        self.read("conf/conf.yaml")
    
    def read(self,file_path : str):
        """Construct an evaluator from a YAML file.
        Args:
            file_path (str): path to the yaml file.
        """
        with open(file_path) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            self.params.update(list(docs)[0][type(self).__name__])


    def describe(self,logs:pd.DataFrame):
        L = []      
        for i in np.unique(logs.epoch):
            if(self.params["shape"] == "last"):
                c = np.array([np.array(i) for i in logs.loc[logs["epoch"] == i, self.params["feature"]].tail(1)])
            else:
                c = np.array([np.array(i) for i in logs.loc[logs["epoch"] == i, self.params["feature"]].sample(self.params["shape"])])
            L.append(c.reshape((-1,)))
            print(c.reshape((-1,)))
        L = np.array(L)
        print("mean : " , L.mean(axis = 0))
        return L


    
        

        


class Archive(object):
    
    def __init__(self,phenotypic_descriptor , *args):
        super(Archive, self).__init__(*args)
        self.phen = phenotypic_descriptor
        self.phenotypes = {}
        self.genotypes = []
        self.novelty = {}
        self.max_size = 1000
        self.distance = np.linalg.norm
        self.archive_spec = {}
        self.read("conf/conf.yaml")
    
    def read(self,file_path : str):
        """Construct an evaluator from a YAML file.
        Args:
            file_path (str): path to the yaml file.
        """
        with open(file_path) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            self.archive_spec.update(list(docs)[0]["archive_spec"])
    
    def clear():
        self.phenotypes.clear()
        self.genotypes.clear()
        self.novelty_reset()
    
    def novelty_reset():
        self.novelty.clear()

    def add(self,genotype: Genotype, logs:pd.DataFrame , *args, **kwargs):
        self.genotypes.append(genotype)
        if(len(self.genotypes) > 1000):
            self.genotypes.pop()
        self.phenotypes[genotype] = self.phen.describe(logs)
        pass

    def update(self,genotypes):
        for genotype in genotypes:
            self.add(genotype)

    def query(self):
        pass

    def search(genotype: Genotype):
        return self.genotypes.index(genotype)

    def get_novelty(genotype: Genotype):
        return self.novelty[genotype]

    def compute_all_novelty(self):
        for i in range(len(self.genotypes)):
            self.compute_novelty(i)


    def compute_novelty(self,id):
        genotype = self.genotypes[id]
        if(genotype in novelty):
            return 
        phenotype = self.phenotypes[genotype]
        #Compute Distances
        dist = {}
        for gen in self.phenotypes:
            if(gen != genotype):
                dist[gen] = self.distance(self.phenotypes[gen] - phenotype)

        #Sort to take the top k-one
        topk = sorted(dist.items(), key = lambda x:x[1])[1:self.archive_spec["novelty_k"]]
        novelty = np.array([i[1] for i in topk]).mean()
        self.novelty[genotype] = novelty

        