#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.misc import imresize, imread


#pyscatter libraries
import tree_database
from scatteringtree import scattering_tree
from nonlinearity import *
from pooling import pooling

import cProfile
import os, struct
from array import array as pyarray
from pylab import *

from sklearn.datasets import fetch_mldata

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from nonlinearity import *





num_train = 1
if len(sys.argv)>1:
    num_train=int(sys.argv[1])

def get_squared_norm(signal):
    #must be an array of any dimension
    signal=signal.flatten().astype(float)
    signal=np.absolute(signal)
    signal=np.power(signal,2)
    e=np.sum(signal)
    return e 

def mnist_to_img(data):
    #data is a 1d array of size 784=28*28
    img_small = data.reshape((28,28))
    resized = imresize(img_small, (32, 32), interp='bilinear')
    flattened = resized.flatten()
    flattened = flattened/(get_squared_norm(flattened)**(1/2))
    return flattened

def run_test():
    run_mnist()

def print_energy(e, amount,size):
    i=0
    print('Propagation protocol:')
    for e in e:
        print('level ',i,': - Energy:', e, ', Amount of signals:', int(amount[i]), ' Signal size:', int(size[i]))
        i+=1

def run_mnist():
    sct = tree_database.simple_1d_filters
    print('Fetching MNIST dataset...')
    mnist = fetch_mldata('MNIST original', data_home= './mnist_dataset')
    labels = mnist.target
    images = mnist.data
    training_input = images[0:num_train]
         
    #Convert to 1024 pixel 1-d array and norm to unity energy
    print('Resample input images...')
    training_input = np.array([mnist_to_img(t) for t in training_input])
    # imgplot = plt.imshow(training_input[0])
    # plt.show()
    print('Scattering training data...')
    scattered_training_output = [sct.transform(x) for x in training_input]
    energies=scattered_training_output[0]['energies']
    amount_of_signals=len(scattered_training_output)
    for i in range(amount_of_signals-1):
        energies=energies+scattered_training_output[i+1]['energies']
    energies=energies/amount_of_signals
    amount_of_props=scattered_training_output[0]['amount']
    size_of_props=scattered_training_output[0]['size']
    print(amount_of_signals,' signals have been propagated.')
    print_energy(energies,amount_of_props,size_of_props)

    
if __name__ == '__main__':
    #cProfile.run('run_test()')
    run_test()