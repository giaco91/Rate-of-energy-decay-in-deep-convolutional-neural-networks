import numpy as np
import time
import os

class Convnet():

    def __init__(self, filters, num_layers):
        self.filters=filters
        self.num_layers=num_layers

    def scatter(self,inputs):
        t0 = time.time()
        if not os.path.exists('mean_times'):
            os.makedirs('mean_times')

        num_inputs=len(inputs)
        energies=np.zeros((self.num_layers+1,num_inputs))
        num_filters=self.filters.num_filters
        for i in range(0,num_inputs):
            energies[0,i]=self.filters.get_squared_norm(inputs[i])
            for k in range(0,num_filters):
                feedback=self.filters.rec(inputs[i],1,self.num_layers,k,num_filters,False,None)
                self.mean_times((2*k+1)/(num_filters*2),time.time()-t0)
                energies[:,i] += feedback[0]+self.filters.rec(None,1,self.num_layers,None,num_filters,feedback[1],feedback[2])[0]
                self.mean_times((2*(k+1))/(num_filters*2),time.time()-t0)
        return energies

    def mean_times(self,ratio,time):
        message=str(100*ratio)+'% - time: '+str(time/60)+' min. Expected end: '+str(time/60/ratio)+' min'
        print(message)
        path='mean_times' + '/' + self.filters.filter_type +'_lay='+str(self.num_layers)+'.txt'
        np.savetxt(path, np.array([ratio,time/60]))       



