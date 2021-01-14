# Illuminated_Learning
Ce travail se base sur l'article de [Davy Smith, Laurissa Tokarchuk et Geraint Wiggins](http://www.eecs.qmul.ac.uk/~laurissa/Laurissas_Pages/Publications_files/SHINE%5B1%5D.pdf)
## Installation
Cloner simplement le repo et installer les "requirements.txt" sur un environnement python 3.5 (a cause de MultiNeat).
Cependant les expériences qui incluent MultiNeat sont minoritaires donc si elles ne vous intéressent pas particulièrement vous n'avez pas besoin de python 3.5 et de MultiNeat.

## Comment lancer une expérience
Toutes les expériences que j'ai écrites sont préfixés par "RUN" et visent toutes a servir un propos dans mon rapport pour en lancer une il suffit de faire :
```bash 
python -m scoop <NOM_EXP>
```
L'expérience créera des lors un dossier dans le dossier "Expériences" ou elle mettra toutes les runs qu'elle effectuera dans un sous dossier dont le nom contiendra une chaine représentant les hyper-paramètres considérés (si il y'a lieu de le faire)
Toutes les expériences effectuées au préalables ont été concervés.

## Comment utiliser le code 

### Classe Experience
Pour écrire une expérience il suffit de créer une classe qui dérive de la classe "Expérience" et de créer dans le dossier conf un fichier de configuration associé.
Ensuite dans cette classe il suffit d'implémenter la méthode "run" et la gestion des fichiers et la création des "resdir" se fera automatiquement avec les méthodes
de la classe mère.

### Code Overview

#### Genome
Pour créer une nouvelle structure de réseau de neurone il suffit de dériver la classe Génôme et de le définir comme on le ferait sur pyTorch : 

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
Cette classe sert a modéliser le processus de calcul de fitness et de sélection des nouveaux individus, donc pour créer un processus de séléction il suffit de dériver
la classe selector et implémenter deux méthodes comme suit :
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
#### Algorithmes 
Le fichier algorithm contient un algorithme NSGA2 prennant en paramètre la classe de genomes a instancier et la classe de sélecteur a utiliser.

## Travail effectué
- Archive SHINE + Tests unitaires.
- Implémentation des méthodes SHINE, MAP-ELITES, FIT, NOVELTY-SEARCH et certaines de leur variantes.
- Etude de l'efficacité de la méthode SHINE suivant les variations des paramètres alpha et beta.
- Etude de l'efficacité de la méthode MAP-Elites suivant les variations de la taille de la grille.
- Implémentation et etude de deux variantes de la méthode SHINE proposée dans l'article basées sur des critères alternatifs et sur du multi-critère.
- Implémentation et étude d'une variante de MAP-Elites basée les coordonnées polaires.



