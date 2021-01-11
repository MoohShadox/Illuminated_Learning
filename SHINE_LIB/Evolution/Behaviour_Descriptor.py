


class Behaviour_Descriptor(object):
    def __init__(self,ind):
        self.ind = ind

    def __len__(self):
        return len(self.ind)

    def __getitem__(self, i):
        return self.ind.bd[i]

    def __eq__(self, other):
        return self.ind[0] == other.ind[0] and self.ind[1] == other.ind[1]

        
    def __repr__(self):
        return 'BD({})'.format(self.ind.bd)