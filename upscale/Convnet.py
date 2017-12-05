import numpy as np

class Convnet():

    def __init__(self, filters, num_layers):
        self.filters=filters
        self.num_layers=num_layers

    def scatter(self,inputs):
        num_inputs=len(inputs)
        energies=np.zeros((self.num_layers+1,num_inputs))
        num_filters=self.filters.num_filters
        for i in range(0,num_inputs):
            energies[0,i]=self.filters.get_squared_norm(inputs[i])
            for k in range(0,num_filters):
                feedback=self.filters.rec(inputs[i],1,self.num_layers,k,num_filters,False,None)
                energies[:,i] += feedback[0]+self.filters.rec(None,1,self.num_layers,None,num_filters,feedback[1],feedback[2])[0]
        return energies


