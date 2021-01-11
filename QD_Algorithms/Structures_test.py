import unittest
from Genotype import *
from Experience import *
import pandas as pd

class Test_Hierarchie(unittest.TestCase):

    def test_genotype_creation(self):
        exp1 = Experience()
        G = Simple_Genotype()
        print(G.get_params().shape)
        df = exp1.simple_evaluation(G)
        print(df)

if __name__ == '__main__':
    unittest.main()