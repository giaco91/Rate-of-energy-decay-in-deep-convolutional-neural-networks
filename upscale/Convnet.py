import numpy as np

class Convnet():

    def __init__(self, filter, num_layers):
        self.filter=filter
        self.num_layers=num_layers

    def scatter(self,inputs):
        num_inputs=len(inputs)
        energies=np.zeros((self.num_layers+1,num_inputs))
        num_filters=self.filter.num_filters
        for i in range(0,num_inputs):
            energies[0,i]=self.filter.get_squared_norm(inputs[i])
            for k in range(0,num_filters):
                energies[:,i] += self.filter.rec(inputs[i],1,self.num_layers,k,num_filters)
        return energies


