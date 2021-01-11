from .fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
import yaml


class Genotype(object):

    def __init__(self, *args):
        super(Genotype, self).__init__(*args)
        self.shape = None
        self.params = None

    def get_params(self):
        return self.params

    def set_params(self,params):
        self.params = params        
    

    def get_action(self,input):
        pass



class Simple_Genotype(Genotype):

    def __init__(self, *args, **kwargs):
        super(Simple_Genotype, self).__init__(*args)

        with open("conf/genotype_specs.yaml","r") as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            self.spec =(list(docs)[0][type(self).__name__])

        self.nn=SimpleNeuralControllerNumpy(self.spec["nb_input"],
                                   self.spec["nb_output"],
                                   self.spec["nb_layers"],
                                   self.spec["nb_neurons_per_layer"])

        self.shape = self.nn.get_parameters().shape
        

    def get_action(self,input):
        return self.nn.predict(input)

    def get_spec(self):
        return self.spec
    
    def set_params(self,params):
        self.nn.set_parameters(params)
        return super().set_params(params)

    def get_params(self):
        return self.nn.get_parameters()

        




        