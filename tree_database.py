from filters.gabortsfm import gabortsfm
from filters.gwt import gwt
from filters.swt import swt1
from filters.swt import swt2
from filters.helper import OutputAllFilter
from pooling import pooling, subsample_pooling
import nonlinearity as nl
import numpy as np
import sys

from scatteringtree import scattering_tree

def get_n2d_filters():
    filter_list=[]
    n=int(sys.argv[3])
    for i in range(n-1):
        filter_list.append([gwt(2, 2, filter_num_rows=2**(n-i), filter_num_columns=2**(n-i))])
    return filter_list

def get_n1d_filters():
    filter_list=[]
    n=int(sys.argv[3])
    for i in range(n-1):
        filter_list.append([swt1(wavelet='db1', level=1, start_level=0, frequency_decreasing_path=False)])
    return filter_list

def get_n2d_poolings():
    pooling_list=[]
    n=int(sys.argv[3])
    for i in range(n-1):
        pooling_list.append(pooling((2,2), np.max))
    return pooling_list

def get_n1d_poolings():
    pooling_list=[]
    n=int(sys.argv[3])
    for i in range(n-1):
        pooling_list.append(pooling(2, np.max))
    return pooling_list


gabor_tree = scattering_tree(
                        [
                 [gabortsfm(3, filter_num_rows=32, filter_num_columns=32)],
                 [gabortsfm(2, filter_num_rows=16, filter_num_columns=16, out_filter='raw')],
                 [gabortsfm(2, filter_num_rows=8, filter_num_columns=8, out_filter='std')],
                 [OutputAllFilter()]
                ],
                [nl.modulus]*2,
                [
                    pooling((2,2), np.max),
                    pooling((2,2), np.max),
                ]
        )



gabor_wavelet_tree = scattering_tree(
                [
                        [gwt(2, 2, filter_num_rows=32, filter_num_columns=32)],
                        [gwt(2, 2, filter_num_rows=16, filter_num_columns=16)],
                        [OutputAllFilter()]
                ],
                [nl.ReLu]*2,
                [
                        pooling((2,2), np.max),
                        pooling((2,2), np.max),
                ]
        )

stationary_wavelet_tree = scattering_tree(
                [
                        [swt2(wavelet='haar', levels=2, start_level=0, frequency_decreasing_path=True, out_filter='raw')],
                        [swt2(wavelet='haar', levels=2, start_level=0, frequency_decreasing_path=True,)],
                        [swt2(wavelet='haar', levels=2, start_level=0, frequency_decreasing_path=True, out_filter='std')],
                        [OutputAllFilter()]
                ],
                [nl.LogSig]*3,
                [
                        subsample_pooling((2,2)),
                        subsample_pooling((2,2)),
                        subsample_pooling((2,2)),
                ]
        )

mixed_tree = scattering_tree(
                [
                        [swt2(wavelet='haar', levels=2, start_level=0),
                         gabortsfm(2, filter_num_rows=32, filter_num_columns=32, out_filter='std')],
                        [gabortsfm(3, filter_num_rows=16, filter_num_columns=16, out_filter=np.square)],
                        [gwt(2,2, filter_num_rows=4, filter_num_columns=4)],
                        [gwt(2,2, filter_num_rows=2, filter_num_columns=2)],
                        [OutputAllFilter()]
                ],
                [
                        nl.ReLu,
                        nl.LogSig
                ],
                [
                        pooling((2,2), np.max),
                        pooling((4,4), np.max),
                        pooling((2,2), np.max),
                 ]
        )

larger_tree = scattering_tree(
                [       
                        [gabortsfm(3, filter_num_rows=64, filter_num_columns=64, out_filter=np.square)],
                        [gwt(2, 2, filter_num_rows=32, filter_num_columns=32)],
                        [gabortsfm(3, filter_num_rows=16, filter_num_columns=16, out_filter=np.square)],
                        [gwt(2, 2, filter_num_rows=8, filter_num_columns=8)],
                        [gwt(2, 2, filter_num_rows=4, filter_num_columns=4)],
                ],
                [nl.ReLu]*2,
                [
                        pooling((2,2), np.max),
                        pooling((2,2), np.max),
                        pooling((2,2), np.max),
                        pooling((2,2), np.max),
                        pooling((2,2), np.max),
                ]
        )

prop_1d_filters = scattering_tree(
                get_n1d_filters(),                    
                [nl.ReLu]*2,
                get_n1d_poolings(),
        )
var_2d_filter = scattering_tree(
               get_n2d_filters(),
                [nl.ReLu]*2,
                get_n2d_poolings(),
        )



