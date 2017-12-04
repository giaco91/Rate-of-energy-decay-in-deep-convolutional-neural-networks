import numpy as np
import time
import sys

from sklearn.datasets import fetch_mldata

import scipy.misc


filter = 'simple_highpass'
num_train = 1
num_layers= 7
folder_name = 'outputs'
if len(sys.argv)>1:
    filter = sys.argv[1]
    if len(sys.argv)>2:
        num_train=int(sys.argv[2])
        if len(sys.argv)>3:
            num_layers=int(sys.argv[3])
            if len(sys.argv)>4:
                folder_name=sys.argv[4]

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

def simple_highpass_filter(fft_signal,support):
    z=np.zeros(1024)
    if support == 1:
        z[1:512]=1
        return np.multiply(fft_signal,z)
    else:
        z[512:1023]=1
        return np.multiply(fft_signal,z)

def raised_cosine_filter(fft_signal,support):
    omega=np.pi/1024
    #calculate the raised cosines
    line=np.linspace(0,1023,num=1024)
    if support==1:
        rc=((np.cos(omega*line)+1)/2)**(1/2)
        rc[0]=0
        rc[512:1024]=0
        return np.multiply(fft_signal,rc)
    if support==2:
        rc_shifted=((np.cos(omega*line + np.pi)+1)/2)**(1/2)
        rc_shifted[0]=0
        rc_shifted[512:1024]=0
        return np.multiply(fft_signal,rc_shifted)
    if support==-1:
        rc=((np.cos(omega*line)+1)/2)**(1/2)
        rc[1023]=0
        rc[0:512]=0
        return np.multiply(fft_signal,rc)

    else:
        rc_shifted=((np.cos(omega*line + np.pi)+1)/2)**(1/2)
        rc_shifted[1023]=0
        rc_shifted[0:512]=0
        return np.multiply(fft_signal,rc_shifted) 

def scatter(signal,support):
    fft_signal=np.fft.fft(signal)
    
    #apply filter
    if filter=='simple_highpass':
        scattered=np.fft.ifft(simple_highpass_filter(fft_signal,support))
    elif filter=='raised_cosine':
        scattered=np.fft.ifft(raised_cosine_filter(fft_signal,support))
    else:
        raise ValueError('The filtertype -' + filter + '- is not defined!')

    
    #apply modulus nonlinearity
    scattered=abs(scattered)
    #no pooling
    return scattered

def simple_highpass_rec(signal,layer,support):
    f=scatter(signal,support)
    if layer==num_layers:
        energies=np.zeros(num_layers+1)
        energies[layer]=get_squared_norm(f)
        return energies
    else:
        energies=simple_highpass_rec(f,layer+1,1)+simple_highpass_rec(f,layer+1,-1)
        energies[layer]+=get_squared_norm(f)
        return energies

def raised_cosine_rec(signal,layer,support):
    f=scatter(signal,support)
    if layer==num_layers:
        energies=np.zeros(num_layers+1)
        energies[layer]=get_squared_norm(f)
        return energies
    else:
        energies=raised_cosine_rec(f,layer+1,1)+raised_cosine_rec(f,layer+1,2)+raised_cosine_rec(f,layer+1,-1)+raised_cosine_rec(f,layer+1,-2)
        energies[layer]+=get_squared_norm(f)
        return energies

def SIMPLE_HIGHPASS(energies, training_input):
    for i in range(0,num_train):
        energies[0,i]+=get_squared_norm(training_input[i])
        energies[:,i] += simple_highpass_rec(training_input[i],1,1) + simple_highpass_rec(training_input[i],1,-1)
    return energies

def RAISED_COSINE(energies, training_input):
    for i in range(0,num_train):
        energies[0,i]+=get_squared_norm(training_input[i])
        energies[:,i] += raised_cosine_rec(training_input[i],1,1) + raised_cosine_rec(training_input[i],1,2)+raised_cosine_rec(training_input[i],1,-1)+raised_cosine_rec(training_input[i],1,-2)
    return energies

def run_mnist():
    print('Filter: ', filter)
    print('Number of inputs: ', num_train)
    print('Number of layers: ', num_layers)

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
    t0 = time.time()
    print('timer started')

    if filter=='simple_highpass':
        energies=SIMPLE_HIGHPASS(energies,training_input)
    elif filter=='raised_cosine':
        energies=RAISED_COSINE(energies,training_input)
    else:
        raise ValueError('The filtertype -' + str(filter) + '- is not defined!')


    averaged_energies=np.average(energies,axis=1)
    print_energy(averaged_energies)
    duration=time.time()-t0
    print('Runtime: ',duration, 'sec')

    #np.savetxt(folder_name + '/generated/l=' + str(cost) + '.csv', np.round(samples, decimals=3), delimiter=',')
    np.savetxt(folder_name + '/' + filter +'_tr='+str(num_train)+'_lay='+str(num_layers)+'.txt', averaged_energies, delimiter=',')  
    print('Energy stored in: ', folder_name + '/' + filter +'_tr='+str(num_train)+'_lay='+str(num_layers)+'.txt')
    
run_mnist()