#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from scipy.misc import imresize, imread
import scipy.misc


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

try:
    from joblib import Parallel, delayed
    from multiprocessing import cpu_count
    n_jobs = cpu_count()
except:
    n_jobs = 1

from nonlinearity import *





verbose = 8
num_train = 1
dimension = 1 #either 1 or 2




def mnist_to_img(data):
    """bring data into square format with dimension being a power of 2"""
    #data is a 1d array of size 784=28*28
    img_small = data.reshape((28,28))
    return scipy.misc.imresize(img_small, (64, 64), interp='bilinear')




def run_test():

    if len(sys.argv)!=4:
        print('You passed no image. Using an image from the MNIST dataset.')
        run_mnist()    
    elif not Path(sys.argv[1]).is_file():
        print('Your first argument is not a path to a valid file. Using an image from the MNIST dataset.')
        run_mnist()  
    else:
        image = scipy.misc.imread(sys.argv[1])
        image=np.average(image, axis=2)
        size=2**int(sys.argv[3])
        if int(sys.argv[2])==2:
            print('preprocessing your signal in 2-dimension...')
            image = scipy.misc.imresize(image, (size, size), interp='bilinear')
            print('Scattering training data...')
            sct = tree_database.var_2d_filters
            out=sct.transform(image)
            energies=out['energies']
            amount_of_props=out['amount']
            size_of_props=out['size']
            print_energy(energies,amount_of_props,size_of_props)
        elif int(sys.argv[2])==1:
            print('preprocessing your signal in 1-dimension...')
            shape=image.shape
            if shape[0]*shape[1]<2**20:
                image=scipy.misc.imresize(image, (2**10, 2**10), interp='bilinear')
            signal=image.flatten() #interprete the image as a 1d signal
            signal=signal[0:1048576]
            sct=tree_database.var_1d_filters
            out=sct.transform(signal)
            energies=out['energies']
            amount_of_props=out['amount']
            size_of_props=out['size']
            print_energy(energies,amount_of_props,size_of_props)
        else:
            raise ValueError('The dimension must be either 1 or 2!')

def print_energy(e, amount,size):
    i=0
    print('Propagation protocol:')
    for e in e:
        print('level ',i,': - Energy:', e, ', Amount of signals:', int(amount[i]), ' Signal size:', int(size[i]))
        i+=1

def run_mnist():
    sct = tree_database.larger_tree
    print('Fetching MNIST dataset...')
    mnist = fetch_mldata('MNIST original', data_home= './mnist_dataset')
    labels = mnist.target
    images = mnist.data
    training_input = images[0:num_train]
         
    #Convert to 32x32 pixel images to satisfy swt2.
    print('Resample input images...')
    training_input = np.array([mnist_to_img(t) for t in training_input])
    # imgplot = plt.imshow(training_input[0])
    # plt.show()
    print('Scattering training data...')
    if n_jobs > 1:
        scattered_training_output = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sct.transform)(inp) for inp in training_input)
    else:
        scattered_training_output = [sct.transform(x) for x in training_input]
        energies=scattered_training_output[0]['energies']
        amount_of_props=scattered_training_output[0]['amount']
        size_of_props=scattered_training_output[0]['size']
        print_energy(energies,amount_of_props,size_of_props)

    
if __name__ == '__main__':
    #cProfile.run('run_test()')
    run_test()