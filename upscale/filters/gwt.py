#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
#from skimage.io import imshow, show
import matplotlib.pyplot as plt
from inspect import isfunction


class gwt:
    """
    A filter bank of Gabor wavelets    
    
    Parameters
    ----------
    num_scales          :   The number of frequency scales of the filter bank
    num_rotations       :   The number of angles of the filter bank
                            The filter bank will contain (num_scales)*(num_rotations) individual filters
    filter_num_rows     :   The dimension of the filters in y-axis, that is, np.size(*,0)
    filter_num_columns  :   The dimension of the filters in x-axis, that is, np.size(*,1)
    out_filter          :   Determines how the output for the feature vector is generated.
                            The following options are available:
                            'std'   :   output is low pass filtered version of input
                            'raw'   :   output is just incoming signal.
                            None    :   no output is generated.
                            func    :   where 'func' is any function that returns an np.array 
                                        and can handle the incoming signal. output is func(signal).
    """
    
    def __init__(self,  num_scales, num_rotations, filter_num_rows=32, filter_num_columns=32, out_filter='std'):
        self._filter_num_rows = filter_num_rows
        self._filter_num_columns = filter_num_columns
        self._num_scales = num_scales
        self._num_rotations = num_rotations
        self._gamma = 0.33
        self._eta = 0.7
        self._sigma_phi = 0.7
        self._sigma_psi = 0.5158
        self._scaling_psi = 0.84
        self._scaling_overall = 1.0
        self._psi_filter_bank = self._generate_psi_filter_bank()
        self._phi_filter = self._generate_phi_filter()
        self._out_filter = out_filter
    

    def apply_filter(self, img, meta=None):
        return _gwt(img, meta, self._num_scales, self._num_rotations, self._phi_filter, self._psi_filter_bank, self._out_filter)
    

    def _generate_psi_filter_bank(self):
        """
        Generates all filters of the filter bank that produce the propagation signals
        """
        psi_filter_bank = [[0]*self._num_rotations]*self._num_scales
        for j in range(0, self._num_scales):
            a = [0]*self._num_rotations
            for ir in range(0, self._num_rotations):
                theta = ir*np.pi/self._num_rotations
                a[ir]=self._generate_psi_single_filter(j+1, theta)
            psi_filter_bank[j]=a
        return psi_filter_bank


    def _generate_psi_single_filter(self, j, theta):
        """
        Generates a single filter of the filter bank that corresponds to frequency scale j 
        and is oriented in the direction of theta
        """
        xi_x, xi_y = self._generate_grid(j, theta)
        for i in range(0, len(xi_x)):
            xi_x[i]=xi_x[i]-self._eta
        for i in range(0, len(xi_y)):
            xi_y[i]=xi_y[i]/self._gamma
        filter = np.exp(-2*((np.pi*self._sigma_psi)**2)*((xi_x**2)+(xi_y**2)))
        filter *= self._scaling_psi*self._scaling_overall*2*self._sigma_psi*np.sqrt(np.pi/self._gamma)/np.sqrt(1+np.exp(-4*(np.pi*self._sigma_psi*self._eta)**2)-np.exp(-6*(np.pi*self._sigma_psi*self._eta)**2))
        return filter
    

    def _generate_phi_filter(self):
        """
        Returns a low pass filter (centered Gaussian) that serves as output generating atom
        (if out_filter='std')
        """
        j = 1
        theta = 0
        xi_x, xi_y = self._generate_grid(j, theta)
        phi_filter = np.exp((-2*(np.pi*self._sigma_phi)**2)*(xi_x**2+xi_y**2))
        phi_filter *= self._scaling_overall*2*self._sigma_phi*np.sqrt(np.pi)
        return phi_filter

    
    def _generate_grid(self, j, theta):
        scale = 2**(j-1)
        x = self._filter_num_rows
        temp_x = list(range(int(np.floor(-self._filter_num_rows/2 + 0.5)),int(np.floor(self._filter_num_rows/2 + 0.5))))
        temp_y = list(range(int(np.floor(-self._filter_num_columns/2 + 0.5)),int(np.floor(self._filter_num_columns/2 + 0.5))))
        temp_x = temp_x/max(np.abs(temp_x))
        temp_y = temp_y/max(np.abs(temp_y))
        xi_x, xi_y = np.meshgrid(temp_x, temp_y)
        xi_x = xi_x/scale
        xi_y = xi_y/scale
        xi_x_temp = xi_x*np.cos(theta) + xi_y*np.sin(theta)
        xi_y_temp = -xi_x*np.sin(theta) +xi_y*np.cos(theta)
        xi_x = xi_x_temp
        xi_y = xi_y_temp
        return [xi_x, xi_y]

def _gwt(img, meta, num_scales, num_rotations, phi_filter, psi_filter_bank, out_filter):
    """
    Parameters
    ----------
        
    img             :	The input signal. Array-like, shape (filter_num_rows, filter_num_columns)
    meta            :	Optional meta data. If 'img' is the output of another gwt, 
                        'meta' contains the scale on which 'img' was generated.
    num_scales      :   The number of scales of the filter bank
    num_rotations   :   The number of rotations of the filter bank
    phi_filter      :   The standard output generating atom
    psi_filter_bank :   The filter bank for propagation signals
    out_filter      :   Determines how the output for the feature vector is generated.
        
    Returns
    -------
    y               :   type dict
                        y['prop']   contains list of propagation signals, 
                                    entries are array-like with shape (filter_num_rows, filter_num_columns)
                        y['out']    contains return value of output generating signal according to out_filter
                        y['meta']   contains list meta information such that 
                                    y['meta'][i] is the number of the scale that filtered y['prop'][i]
        
    """
    prop = []
    # imgplot = plt.imshow(img)
    # plt.show()
    inp_img = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    # imgplot = plt.imshow(np.absolute(inp_img))
    # plt.show()
    #output for the feature vector
    if out_filter==None:
        out = []
    elif type(out_filter)==str and out_filter=='raw':
        out = [img]
    elif type(out_filter)==str and out_filter=='std':
        if(phi_filter.size > inp_img.size):
            raise NameError('Wavelet_Transform::ApplyTransform: Filter size larger than input image size')
        out = []
        oup_img = inp_img*phi_filter
        oup_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(oup_img)))
        out = [oup_img]
    elif hasattr(out_filter,'__call__'):
        out = [out_filter(img)]
    else:
        print(out_filter, " is not a valid choice for output generation")
        exit(1)
        
    #scales of higher frequency than the one from which 'img' stems will not be considered
    if meta is not None and 'scale' in meta:
        current_scale = min(meta['scale'], num_scales)
    else:
        current_scale = num_scales

    #output for propagation signals
    new_meta = []
    for i in range(0, current_scale):
        for j in range(0, num_rotations):
            if(psi_filter_bank[i][j].size > inp_img.size):
                raise NameError('Wavelet_Transform::ApplyTransform: Filter size larger than input image size')
            prop_img = inp_img*psi_filter_bank[i][j]
            prop_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(prop_img)))
            prop.extend([prop_img])
            m = {'scale': i+1}
            new_meta.extend([m])

    return {'prop': prop, 'out': out, 'meta': new_meta}


##if __name__ == '__main__':
##    """
##    Plot gabor filter bank.
##    """
##    
##    tf = gwt(num_scales=4, num_rotations=4, filter_num_rows=128, filter_num_columns=128)
##
##    psifilters = [val for one_filter_bank in tf._psi_filter_bank for val in one_filter_bank]
##    phifilter = np.reshape(tf._phi_filter,(len(tf._phi_filter),len(tf._phi_filter)))
##    fig, axes = plt.subplots(4, 4, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
##    fig.subplots_adjust(hspace=0.25, wspace=0.25)
##
##    for ax, s in zip(axes.flat, psifilters):
##        im = ax.imshow(s, interpolation='nearest')
##    plt.show()







                      
