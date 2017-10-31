"""
This module implements functions for down-sampling by pooling.
"""

import numpy as np

class pooling:
    """Downsample a 1D or 2D signal by taking pool_size samples together (average or max).

    Parameters
    ----------
    
    pool_size   :   How many values to combine into one.
                    For one dimensional signals this is expected to be a integer.
                    For multi dimensional signals this must be a tuple of integers, one value per dimension.
                    
    pooling_function :   Might be one of: np.mean, np.max, ...
    
    
    Examples
    --------
    
    # Apply pooling function to an image.
    p = pooling((2,2), np.mean)
    pooled = p.apply_pooling(image)
    
    # Apply pooling function to audio.
    p = pooling(2, np.mean)
    pooled = p.apply_pooling(wave)
    
    """
    
    def __init__(self, pool_size, pooling_function):
        self._pool_size = pool_size
        self._pooling_function = pooling_function
        
    def apply_pooling(self, signal):
        """The input signal gets down-sampled.
        """
        return _apply_pooling(signal, self._pool_size, self._pooling_function)


def _apply_pooling(signal, pool_size, poolingfunc):
    """Downsample a 1D or 2D signal by taking pool_size samples together (average or max).
    
    Parameters
    ----------
    
    pool_size   :   How many values to combine into one.
                    For one dimensional signals this is expected to be a integer.
                    For multi dimensional signals this must be a tuple of integers, one value per dimension.
                    
    poolingfunc :   Might be one of: np.mean, np.max, ...
    
    """

    num_split = np.array(signal.shape) / np.array(pool_size)

    
    # split by columns
    cols = np.split(signal, num_split[0], 0)
    pooled = poolingfunc(cols, 1)

    if len(signal.shape) > 1:
        rows = np.split(pooled, num_split[1], 1)
        pooled = poolingfunc(rows, 2)
        pooled = pooled.transpose()

    return pooled



class subsample_pooling:
    """Downsample a 1D or 2D signal by subsampling with sampling ratio sample_size.
    
    Parameters
    ----------
    sample_size :   The sampling ratio. If 1D signals are expected, sample_size is 
                    an integer. If 2D signals are expected, sample_size is a 2-tuple
                    where sample_size[0] is the sampling ratio of rows and where
                    sample_size[1] is the sampling ratio of columns.
                    
    Examples
    --------
                    
    # Apply subsample_pooling  to an image.
    p = subsample_pooling((2,2))
    pooled = p.apply_pooling(image)
                    
    # Apply pooling function to audio.
    p = subsample_pooling(2)
    pooled = p.apply_pooling(wave)
        
    """
    
    def __init__(self, sample_size):
        self._sample_size = sample_size

    def apply_pooling(self, signal):
        """Downsample a 1D or 2D signal by subsampling
        
        Parameters
        ----------
        sample_size :   The sampling ratio. If signal is 1D, sample_size is
                        an integer. If signal is 2D, sample_size is a 2-tuple
                        where sample_size[0] is the sampling ratio of rows and where
                        sample_size[1] is the sampling ratio of columns.
        signal      :   The signal that is being subsampled.
        
        Returns
        -------
        result      :   The subsampled signal.

        """
        s = self._sample_size
        
        if len(s)==2:
            #signal is 2D
            result = signal[::s[0],::s[1]]
        else:
            #signal is 1D
            result = signal[::s]

        return result















