#coding: utf-8



import numpy as np
import pywt

class dh:

    
    def __init__(self, scale):
        self.scale = scale
        
    def apply_filter(self, signal):

        return DH(signal, self.scale)


def DH(signal, scale=9):
    """
    Parameters
    ----------
    signal  :	The input signal.
    scale   :   Can be: 1,3,9, with the corresponding dyadic intervalls: [2^(scale*(n-1)),2^(scale*n)]
    """



    prop = dyadic_highpass(signal, scale) 

    """
        prop = [f1, f2, .., f]
    """

    return {'prop': prop}



    
def dyadic_highpass(signal,scale):
    fft_signal = np.fft.fft(signal)
    prop=[]
    for i in range(0,int(9/scale)):
        z1=np.zeros(1024)
        z2=np.zeros(1024)
        z1[2**(i*scale):2**((i+1)*scale)]=1
        z2[1024-2**((i+1)*scale):1024-2**(i*scale)]=1
        f1=np.multiply(fft_signal,z1)#positive frequencies
        f2=np.multiply(fft_signal,z2)#negative frequencies
        prop.extend([np.fft.ifft(f1)])
        prop.extend([np.fft.ifft(f2)])
    return prop
        


    

