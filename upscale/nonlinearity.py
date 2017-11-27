"""
This module implements several non-linearities
"""

import numpy as np
from numpy import max, real, imag


def ReLu(x):
    """
    Rectified linear unit non-linearity
    """
    z = np.zeros(x.shape)
    return np.max([z, real(x)], 0) + 1j*np.max([z, imag(x)], 0)

def modulus(x):
    """
    Modulus non-linearity
    """
    return abs(x)

def TanHyp(x):
    """
    Hyperbolic tangent non-linearity"
    """
    return np.tanh(real(x))+1j*np.tanh(imag(x))

def LogSig(x):
    """
    Logistic sigmoid non-linearity
    """
    def sig(x):
        return 1/(1+np.exp(-x))
    
    return sig(real(x))+1j*sig(imag(x))

