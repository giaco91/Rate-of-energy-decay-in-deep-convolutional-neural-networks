#coding: utf-8
"""
This module contains wrapper classes around
one and two dimensional stationary wavelet transforms.
"""


import numpy as np
import pywt

class swt1:
    """1D stationary wavelet transform.
        
    Parameters
    ----------
    wavelet                     :   The wavelet type.
    level                       :   The number of scales.
    start_level                 :   Begin at this scale.
    frequency_decreasing_path   :   Reduce overhead by cutting off some branches of the scattering tree.
    out_filter                  :   Determines how the output for the feature vector is generated.
                                    The following options are available:
                                    'std'   :   output is approximation cA of last level.
                                    'raw'   :   output is just incoming signal.
                                    None    :   no output is generated.
                                    func    :   where 'func' is any function that returns an np.array and can handle
                                                the incoming signal. output is func(signal).

    """
    
    def __init__(self, wavelet='db1', level=1, start_level=0, frequency_decreasing_path=False, out_filter='std'):
        self._wavelet = wavelet
        self._level = level
        self._start_level = start_level
        self._frequency_decreasing_path = frequency_decreasing_path
        self._out_filter = out_filter
        
    def apply_filter(self, signal, meta=None):
        """ Generate the output of the wavelet transform.
        Parameters
        ----------
        signal  :   one-dimensional, list-like, length must be divisible by 2**self._level 
        meta    :   The optional meta data. If 'signal' is the output of another swt1, 'meta' contains
                    the number of the swt-level on which 'signal' was generated.
        
        Returns
        -------
        y       :   type dict
                    y['prop']   contains list of propagation signals,
                                entries are list-like with length identical to length of signal
                    y['out']    contains return value of output generating signal according to out_filter
                    y['meta']   contains list meta information such that
                                y['meta'][i] is the number of the swt-level that filtered y['prop'][i]

        """
        return _swt1(signal, meta, self._wavelet, self._level, self._start_level, self._frequency_decreasing_path, self._out_filter)


def _swt1(signal, meta, wavelet, level=1, start_level=0, frequency_decreasing_path=False, out_filter='std'):
    """
    Parameters
    ----------
    signal  :	The input signal.
    wavelet :	The wavelet type.
    meta    :	Optional meta data. If 'signal' is the output of another swt1, 'meta' contains
                the number of the swt-level on which 'signal' was generated.
                
    Returns
    -------
    y       :   type dict
                y['prop']   contains list of propagation signals,
                            entries are list-like with length identical to length of signal
                y['out']    contains return value of output generating signal according to out_filter
                y['meta']   contains list meta information such that
                            y['meta'][i] is the number of the swt-level that filtered y['prop'][i]
    

    """

    tf = pywt.swt(signal, pywt.Wavelet(wavelet), level=level, start_level=start_level)

    """
        tf = [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]
    """

    # Get the maximum scale to propagate.
    if meta is not None and 'scale' in meta:
        # We get the scale from the previous filter.
        max_scale = meta['scale']
    else:
        # The previous filter was not polite enouth to provide us with that information.
        max_scale = len(tf)

    prop = []
    out = []
    new_meta = []

    for (cA, cD), scale in zip(tf, range(len(tf), 0, -1)):
        # scale takes values in: len(tf), ..., 3,2,1
        
        if not frequency_decreasing_path or scale <= max_scale:
            prop.append(cD)
            m = {'scale': scale}
            new_meta.append(m)
        

    #generate output depending on settings
    if out_filter==None:
        #no output is generated
        out = []
    elif type(out_filter)==str and out_filter=='raw':
        #original signal is passed to output
        out = [signal]
    elif type(out_filter)==str and out_filter=='std':
        #output cA of last level.
        out = [cA]
    elif hasattr(out_filter,'__call__'):
        #whatever filter has been given as out_filter is applied to the signal
        out = [out_filter(signal)]
    else:
        #something went wrong
        print(out_filter, " is not a valid choice for output generation")
        exit(1)


    return {'prop': prop, 'out': out, 'meta': new_meta}


class swt2:
    """2D stationary wavelet transform.
   
    Parameters
    ----------
    wavelet                     :   The wavelet type.
    levels                      :   The number of scales.
    start_level                 :   Begin at this scale.
    frequency_decreasing_path   :   Reduce overhead by cutting off some branches of the scattering tree.
    out_filter                  :   Determines how the output for the feature vector is generated. 
                                    The following options are available:
                                    'std'   :   output is approximation cA of last level.
                                    'raw'   :   output is just incoming signal.
                                    None    :   no output is generated.
                                    func    :   where 'func' is any function that returns an np.array and can handle 
                                                the incoming signal. output is func(signal).
    
    """
    
    def __init__(self, wavelet='haar', levels=1, start_level=0, frequency_decreasing_path=False, out_filter='std'):
        self._wavelet = wavelet
        self._level = levels
        self._start_level = start_level
        self._frequency_decreasing_path = frequency_decreasing_path
        self._out_filter = out_filter
    
        
    def apply_filter(self, signal, meta=None):
        """
        Generate the output of the wavelet transform.
    
        Parameters
        ----------
        signal  :   two-dimensional array-like, dimensions must be divisible by 2**self._level
        meta    :   Optional meta data. If 'signal' is the output of another swt2, 'meta' contains
                the number of the swt-level on which 'signal' was generated.
                
        Returns
        -------
        y       :   type dict
                    y['prop']   contains list of propagation signals,
                                entries are array-like with shape identical to shape of signal
                    y['out']    contains return value of output generating signal according to out_filter
                    y['meta']   contains list meta information such that
                                y['meta'][i] is the number of the swt-level that filtered y['prop'][i]
        """
        return _swt2(signal, meta, self._wavelet, self._level, self._start_level, self._frequency_decreasing_path, self._out_filter)
    

def _swt2(signal, meta, wavelet, level, start_level=0, frequency_decreasing_path=False, out_filter='std'):
    """
    Parameters
    ----------
    signal      :	The input signal.
    wavelet     :	The wavelet type.
    meta        :	Optional meta data. If 'signal' is the output of another swt2, 'meta' contains
                    the number of the swt-level on which 'signal' was generated.
    out_filter  :   Determines how the output for the feature vector is generated.
    
    Returns
    -------
    y       :   type dict
                y['prop']   contains list of propagation signals,
                            entries are array-like with shape (filter_num_rows, filter_num_columns)
                y['out']    contains return value of output generating signal according to out_filter
                y['meta']   contains list meta information such that
                            y['meta'][i] is the number of the swt-level that filtered y['prop'][i]
    """
    
    tf = pywt.swt2(signal, pywt.Wavelet(wavelet), level=level, start_level=start_level)

    """    
        tf =
                [
                    (cA_n,
                        (cH_n, cV_n, cD_n)
                    ),
                    (cA_n+1,
                        (cH_n+1, cV_n+1, cD_n+1)
                    ),
                    ...,
                    (cA_n+level,
                        (cH_n+level, cV_n+level, cD_n+level)
                    )
                ]
    """

    
    # Get the maximum scale to propagate.
    if meta is not None and 'scale' in meta:
        # We get the scale from the previous filter.
        max_scale = meta['scale']
    else:
        # The previous filter was not polite enouth to provide us with that information.
        max_scale = len(tf)

    prop = []
    out = []
    new_meta = []


    def _rescale(img,j):
        return img*(2**(-2*j))
 
    j = 0
    for (cA, cHVD), scale in zip(tf, range(len(tf), 0, -1)):
        # scale takes values in: len(tf), ..., 3,2,1
        
        if not frequency_decreasing_path or scale <= max_scale:
            
            cHVD = [ _rescale(c,j) for c in cHVD ]
            
            prop.extend(cHVD)
            m = {'scale': scale}
            new_meta.extend([m]*len(cHVD))
            
            j += 1
            cA_out = _rescale(cA,j)
    
    
    #generate output depending on settings
    if out_filter==None:
        #no output is generated
        out = []
    elif type(out_filter)==str and out_filter=='raw':
        #original signal is passed to output
        out = [signal]
    elif type(out_filter)==str and out_filter=='std':
        #output cA of last level.
        out = [cA_out]
    elif hasattr(out_filter,'__call__'):
        #whatever filter has been given as out_filter is applied to the signal
        out = [out_filter(signal)]
    else:
        #something went wrong
        print(out_filter, " is not a valid choice for output generation")
        exit(1)

    return {'prop': prop, 'out': out, 'meta': new_meta}

