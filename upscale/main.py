import numpy as np
import time
import sys

from sklearn.datasets import fetch_mldata

import scipy.misc



num_train = 1
num_layers= 10
folder_name = 'outputs'
#num_proc=1
if len(sys.argv)>1:
    num_train=int(sys.argv[1])
    num_layers=int(sys.argv[2])
    if len(sys.argv)>3:
        folder_name=sys.argv[3]

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
    resized = scipy.misc.imresize(img_small, (32, 32), interp='bilinear')
    flattened = resized.flatten()
    flattened = flattened/(get_squared_norm(flattened)**(1/2))
    return flattened


def print_energy(energies):
    print('Propagation protocol:')
    for i in range(0,len(energies)):
        print('level ',i,': - Energy:', energies[i])

def dyadic_highpass(fft_signal,support):
    z=np.zeros(1024)
    if support == 1:
        z[1:512]=1
        f=np.multiply(fft_signal,z)
        return np.fft.ifft(f)
    else:
        z[512:1023]=1
        f=np.multiply(fft_signal,z)
        return np.fft.ifft(f)

def scatter(signal,support):
    fft_signal=np.fft.fft(signal)
    #apply filter
    scattered=dyadic_highpass(fft_signal,support)
    #apply modulus nonlinearity
    scattered=abs(scattered)
    #no pooling
    return scattered

def propagation(signal,layer,support):
    # if proc==1:
    #     print('PROCESSOR IN PROPAGATION')
    #     proc=0
    # elif proc>=2: 
    #     proc=proc/2
    f=scatter(signal,support)
    if layer==num_layers:
        energies=np.zeros(num_layers+1)
        energies[layer]+=get_squared_norm(f)
        return energies
    else:
        energies=propagation(f,layer+1,1)+propagation(f,layer+1,-1)
        energies[layer]+=get_squared_norm(f)
        return energies



def run_mnist():
    #set up scattering tree
    #sct = tree_database.simple_1d_filters

    #read in data
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

    energies=np.zeros((num_layers+1,num_train))
    print('Scattering training data...')
    #processor analysis
    #num_processors=num_proc
    #num_proc_per_prop=2**np.floor(np.log2(num_processors/num_train))
    #print('processors per input',num_proc_per_prop)
    #num_processors=num_train*num_proc_per_prop
    #print('processors needed',num_processors)
    t0 = time.time()
    print('timer started')
    for i in range(0,num_train):
        energies[0,i]+=get_squared_norm(training_input[i])
        energies[:,i] += propagation(training_input[i],1,1) + propagation(training_input[i],1,-1)

    averaged_energies=np.average(energies,axis=1)
    print_energy(averaged_energies)
    duration=time.time()-t0
    print('Runtime: ',duration, 'sec')

    #np.savetxt(folder_name + '/generated/l=' + str(cost) + '.csv', np.round(samples, decimals=3), delimiter=',')
    np.savetxt(folder_name + '/simple_highpass'+str(num_layers)+'.txt', averaged_energies, delimiter=',')
    
run_mnist()