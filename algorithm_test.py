import unittest
import pandas as pd
from algorithms import *
from Archive import *
from QD_Algorithms.Genotype import Simple_Genotype
import numpy as np
class Test_Hierarchie(unittest.TestCase):
    
    #def test_function_generator(self):
    #    a,b,c = simple_evaluation_generator("dist_obj",10000, 10,Simple_Genotype)
    #    print(a)
    #    print(b)
    #    arr = np.zeros(b)
    #    print(a(arr))
    #    pass
#
    #def test_mo_generator(self):
    #    a,b,c = mo_evaluation_generator(["dist_obj","exit_reached"],10000, 10,Simple_Genotype)
    #    print(a)
    #    print(b)
    #    arr = np.zeros(b)
    #    print(a(arr))
#
    def test_evolutionnary_algorithm(self):
        ea = Evolutionnary_Algorithm(Simple_Genotype)
        ea.initialise()
        ea.create_population()
        ea.evolve()

    #def test_archive(self):
    #    e = Experience()
    #    pd = Phenotypic_Descriptor()
    #    archive = Archive(pd)
    #    print(archive)
    #    for i in range(10):
    #        G = Simple_Genotype()
    #        logs = e.get_logs(G)
    #        archive.add(G,logs)
    #    for j in range(10):
    #        archive.compute_novelty(j)
    #    print(archive.novelty)

        

if __name__ == '__main__':
    unittest.main()
