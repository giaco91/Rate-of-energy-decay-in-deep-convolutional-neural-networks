#coding: utf-8



import numpy as np
import pywt

class rc:

    
    def __init__(self, omega):
        self.omega = omega
        
    def apply_filter(self, signal):

        return RC(signal, self.omega)


def RC(signal, omega=0.01):
    """
    Parameters
    ----------
    signal  :	The input signal.
    scale   :   Can be: 1,3,9, with the corresponding dyadic intervalls: [2^(scale*(n-1)),2^(scale*n)]
    """



    prop = raised_cosine(signal, omega) 

    """
        prop = [f1, f2, .., f]
    """

    return {'prop': prop}



    
def raised_cosine(signal,omega):
    fft_signal = np.fft.fft(signal)
    prop=[]
    #calculate the raised cosinuses
    line=np.linspace(0,1023,num=1024)
    rc=((np.cos(omega*line)+1)/2)**(1/2)
    rc_shifted=((np.cos(omega*line + np.pi)+1)/2)**(1/2)
    #print(rc**2+rc_shifted**2)
    
    #make highpass
    rc[0]=0
    rc[1023]=0
    rc_shifted[0]=0
    rc_shifted[1023]=0
    #print(rc**2+rc_shifted**2)
    
    #divide in positive and negative regimes
    rc1=np.copy(rc)
    rc2=np.copy(rc_shifted)
    rc3=np.copy(rc)
    rc4=np.copy(rc_shifted)
    rc1[512:1024]=0
    rc2[512:1024]=0
    rc3[0:512]=0
    rc4[0:512]=0
    #print(rc1**2+rc2**2+rc3**2+rc4**2)

    #multiply signal in freq domain with filters and transform back
    prop.extend([np.fft.ifft(np.multiply(fft_signal,rc1))])
    prop.extend([np.fft.ifft(np.multiply(fft_signal,rc2))])
    prop.extend([np.fft.ifft(np.multiply(fft_signal,rc3))])
    prop.extend([np.fft.ifft(np.multiply(fft_signal,rc4))])
    return prop


        