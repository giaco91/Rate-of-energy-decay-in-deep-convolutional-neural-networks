import numpy as np
import time
import sys
import os

from sklearn.datasets import fetch_mldata

import scipy.misc
from Convnet import Convnet
from Filter_types import *


filter_type = 'simple_highpass'
num_inputs = 1
num_layers= 7
folder_name = 'outputs'
if len(sys.argv)>1:
    filter_type = sys.argv[1]
    if len(sys.argv)>2:
        num_inputs=int(sys.argv[2])
        if len(sys.argv)>3:
            num_layers=int(sys.argv[3])
            if len(sys.argv)>4:
                folder_name=sys.argv[4]

if not os.path.exists(folder_name):
    os.makedirs(folder_name)



def mnist_to_img(data):
    #data is a 1d array of size 784=28*28
    img_small = data.reshape((28,28))
    resized = scipy.misc.imresize(img_small, (32, 32), interp='bilinear')
    flattened = resized.flatten()
    #make signal length uneven
    z=np.zeros(1025)
    z[0:1024]=flattened
    flattened = z/(get_squared_norm(z)**(1/2))
    return flattened

def get_squared_norm(signal):
    signal=np.absolute(signal)
    signal=signal.flatten().astype(float)
    signal=np.power(signal,2)
    e=np.sum(signal)
    return e

def print_energy(energies):
    print('Propagation protocol:')
    for i in range(0,len(energies)):
        print('level ',i,': - Energy:', energies[i])

def get_filters(filter_type):
    if filter_type=='simple_highpass':
        return Simple_highpass()
    if filter_type=='raised_cosine':
        return Raised_cosine()
    if filter_type=='wavelet_rect':
        return Wavelet_rect()
    else:
        raise ValueError('The filtertype "' + filter_type + '" is not defined!')

def run_mnist():
    print('Filter: ', filter_type)
    print('Number of inputs: ', num_inputs)
    print('Number of layers: ', num_layers)

    #read in data
    print('Fetching MNIST dataset...')
    mnist = fetch_mldata('MNIST original', data_home= './mnist_dataset')
    images = mnist.data
    inputs = images[0:num_inputs]
         
    #Convert to 1024 pixel 1-d array and norm to unity energy
    print('Resample input images...')
    inputs = np.array([mnist_to_img(t) for t in inputs])

    print('Scattering training data...')
    t0 = time.time()
    print('timer started')

    filters=get_filters(filter_type)
    net=Convnet(filters,num_layers)
    energies=net.scatter(inputs)

    averaged_energies=np.average(energies,axis=1)
    print_energy(averaged_energies)
    duration=time.time()-t0
    print('Runtime: ',duration, 'sec')

    #np.savetxt(folder_name + '/generated/l=' + str(cost) + '.csv', np.round(samples, decimals=3), delimiter=',')
    np.savetxt(folder_name + '/' + filter_type +'_tr='+str(num_inputs)+'_lay='+str(num_layers)+'.txt', averaged_energies, delimiter=',')  
    print('Energy stored in: ', folder_name + '/' + filter_type +'_inp='+str(num_inputs)+'_lay='+str(num_layers)+'.txt')
    


run_mnist()
 
















