# Illuminated_Learning
This is based on the article of [Davy Smith, Laurissa Tokarchuk et Geraint Wiggins](http://www.eecs.qmul.ac.uk/~laurissa/Laurissas_Pages/Publications_files/SHINE%5B1%5D.pdf)
## Installation
Clone the repo and install the "requirements.txt" on a python 3.5 environment (because of MultiNeat). However, experiences that include MultiNeat are in the minority so if you are not particularly interested you don't need python 3.5 and MultiNeat.

## How to run an experiment 
All the experiments we have written are prefixed by "RUN" and all aim to serve a purpose in my report. To launch one just do :
```bash 
python -m scoop <NOM_EXP>
```
The experiment will then create a folder in the "Experience" folder where it will put all the runs it performs in a sub-folder whose name will contain a string representing the considered hyper-parameters (if any). All the experiments performed beforehand have been kept.

## How to use the code 

### Class "Experience"

To write an experiment, you just have to create a class that derives from the "Experience" class and to create in the conf folder an associated configuration file. Then in this class you just have to implement the "run" method  and the file management so the creation of the "resdir" will be done automatically with the methods of the parent class.

### Code Overview

#### Genome
To create a new neural network structure, just derive the Genome class and define it as you would do on pyTorch :

```python
class DNN(Genome):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.max_action = kwargs["max_action"]
        self.l1 = nn.Linear(kwargs["nb_input"],10)
        self.l2 = nn.Linear(10,10)
        self.l3 = nn.Linear(10,10)
        self.out = nn.Linear(10,kwargs["nb_output"])

               
    def predict(self, x):
        x = torch.tensor(x)
        x = self.sigm(self.l1(x))
        x = self.sigm(self.l2(x))
        x = self.sigm(self.l3(x))

        x = self.out(x)
        x = torch.tanh(x)
        x = x.detach().numpy()
        return x*self.max_action
```
#### Selector
This class is used to model the process of fitness calculation and the selection of new individuals, so to create a selection process it is enough to derive the selector class and implement two methods as follows:

```python

class Selector_SHINE_COL(Selector):
    
    def __init__(self, **kwargs):
        self.archive = Shine_Archive_COL(600,600,alpha=kwargs["alpha"],beta=kwargs["beta"])
        self.grid = Grid(kwargs["grid_min_v"],kwargs["grid_max_v"],kwargs["dim_grid"])
        
    ### Met a jour le selector avec la nouvelle génération
    def update_with_offspring(self, offspring):
        for ind in offspring:
            self.grid.add(ind)
        self.archive.update_offspring(offspring)
        pass
    
    #Calcul les fitness
    def compute_objectifs(self, population):
        for i in population:
            n = self.archive.search(Behaviour_Descriptor(i))
            if(n!= None and len(n.val) > 0):
                i.fitness.values = (n.level  ,len(n.val) )
            else:
                i.fitness.values = (np.inf,self.archive.beta,len(n.val))
        pass
    #Sélectionne des individus
    def select(self, pq, mu):
        return tools.selNSGA2(pq,mu)

```
#### Algorithms 
The algorithm file contains an NSGA2 algorithm taking in parameter the genome class and the selector class.

## Work done 
- SHINE Archive + Unit Tests.
- Implementation of SHINE, MAP-ELITES, FIT, NOVELTY-SEARCH methods and some of their variants.
- Study of the efficiency of the SHINE method according to the variations of the alpha and beta parameters.
- Study of the efficiency of the MAP-Elites method according to the variations of the grid size.
- Implementation and study of two variants of the SHINE method proposed in the article based on alternative criteria and multi-criteria.
- Implementation and study of a variant of MAP-Elites based on polar coordinates.



