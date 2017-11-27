#coding: utf-8
import numpy as np
from inspect import isfunction


class gabortsfm:
    """
    A filter bank of two-dimensional Gaussians of identical variance 
    whose centers form a two-dimensional grid in the frequency domain
    
    Parameters
    ----------
    num_coronas         :   The number of coronas of the grid. The grid is quadratic with side length 2*num_coronas+1
    filter_num_rows     :   The dimension of the filters in y-axis, that is, np.size(*,0)
    filter_num_columns  :   The dimension of the filters in x-axis, that is, np.size(*,1)
    out_filter          :   Determines how the output for the feature vector is generated.
                            The following options are available:
                            'std'   :   output is low pass filtered version of input
                            'raw'   :   output is just incoming signal.
                            None    :   no output is generated.
                            func    :   where 'func' is any function that returns an np.array and can handle the incoming signal. 
                                        output is func(signal).
                            

    """
    
    def __init__(self,  num_coronas, filter_num_rows=32, filter_num_columns=32, out_filter='std'):
        self._filter_num_rows = filter_num_rows
        self._filter_num_columns = filter_num_columns
        self._num_coronas = num_coronas
        self._sigma_psi = 0.5*(num_coronas+1)    #rule of thumb
        self._out_filter = out_filter
        self._psi_filter_bank = self._generate_psi_filter_bank()
        self._phi_filter = self._generate_phi_filter()


    def apply_filter(self, img, meta=None):
        """
        Wrapper for _gabortsfm
        
        Parameters
        ----------
        img     :	The input signal. Array-like, shape (filter_num_rows, filter_num_columns)
        meta    :	Optional meta data. If 'img' is the output of another swt2, 
                    'meta' contains the number of the corona on which 'img' was generated.
        
        Returns
        -------
        y       :   type dict
                    y['prop']   contains list of propagation signals, 
                                entries are array-like with shape (filter_num_rows, filter_num_columns)
                    y['out']    contains return value of output generating signal according to out_filter
                    y['meta']   contains list meta information such that 
                                y['meta'][i] is the number of the corona that filtered y['prop'][i]

        """
        
        return _gabortsfm(img, meta, self._num_coronas, self._phi_filter, self._psi_filter_bank, self._out_filter)



    def _generate_psi_filter_bank(self):
        """
        Generates all filters of the filter bank that produce the propagation signals
        """
        psi_filter_bank = [[0]*(2*self._num_coronas+1)]*(2*self._num_coronas+1)
        for i in range(0, len(psi_filter_bank)):
            a = [0]*len(psi_filter_bank[i])
            for j in range(0, len(psi_filter_bank)):
                a[j]=self._generate_psi_single_filter(j-self._num_coronas, i-self._num_coronas)
            psi_filter_bank[i]=a
        return psi_filter_bank


    def _generate_psi_single_filter(self, xp, yp):
        """
        Generates a two-dimensional Gaussian with center at (xp, yp) for a propagation signal
        """
        xi_x, xi_y = self._generate_grid(xp, yp)
        filter = (2*np.pi/self._sigma_psi)*np.exp(-2*((np.pi*self._sigma_psi)**2)*((xi_x**2)+(xi_y**2)))
        return filter
    
    
    def _generate_phi_filter(self):
        """
        Returns a low pass filter (centered Gaussian) that serves as output generating atom
        (if out_filter='std')
        """
        xi_x, xi_y = self._generate_grid(0, 0)
        return (2*np.pi/self._sigma_psi)*np.exp(-2*((np.pi*self._sigma_psi)**2)*((xi_x**2)+(xi_y**2)))


    def _generate_grid(self, xp, yp):
        x = self._filter_num_rows
        temp_x = list(range(int(np.floor(-self._filter_num_rows/2 + 0.5)),int(np.floor(self._filter_num_rows/2 + 0.5))))
        temp_y = list(range(int(np.floor(-self._filter_num_columns/2 + 0.5)),int(np.floor(self._filter_num_columns/2 + 0.5))))
        temp_x = temp_x/max(np.abs(temp_x))
        temp_y = temp_y/max(np.abs(temp_y))
        xi_x, xi_y = np.meshgrid(temp_x, temp_y)
        for k in range(0, xi_x.shape[0]):
            for l in range(0, xi_x.shape[1]):
                xi_x[k,l]=xi_x[k,l]-xp/(self._num_coronas+1)
        for k in range(0, xi_y.shape[0]):
            for l in range(0, xi_y.shape[1]):
                xi_y[k,l]=xi_y[k,l]-yp/(self._num_coronas+1)
        return [xi_x, xi_y]




def _gabortsfm(img, meta, num_coronas, phi_filter, psi_filter_bank, out_filter):
    """
    Parameters
    ----------
        
    img             :	The input signal. Array-like, shape (filter_num_rows, filter_num_columns)
    meta            :	Optional meta data. 
                        If 'img' is the output of another gabortsfm, 
                        'meta' contains the number of the corona on which 'img' was generated.
    num_coronas     :   The number of coronas of the filter bank
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
                                    y['meta'][i] is the number of the corona that filtered y['prop'][i]
    
    """

    prop = []
    out = []
    inp_img = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    
    #output for the feature vector
    if out_filter==None:
        out = []
    elif type(out_filter)==str and out_filter=='raw':
        out = [img]
    elif type(out_filter)==str and out_filter=='std':
        if(phi_filter.size > inp_img.size):
            raise NameError('Wavelet_Transform::ApplyTransform: Filter size larger than input image size')
        oup_img = inp_img*phi_filter
        oup_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(oup_img)))
        out = [oup_img]
    elif hasattr(out_filter,'__call__'):
        out = [out_filter(img)]
    else:
        print(out_filter, " is not a valid choice for output generation")
        exit(1)

    

    #coronas of higher frequency than the one from which 'img' stems will not be considered
    if meta is not None and 'corona' in meta:
        current_scale = min(meta['corona'], num_coronas)
    else:
        current_scale = num_coronas

    #output for propagation signals
    new_meta = []
    for i in range(num_coronas-current_scale, num_coronas+current_scale+1):
        for j in range(num_coronas-current_scale, num_coronas+current_scale+1):
            if(psi_filter_bank[i][j].size > inp_img.size):
                raise NameError('Wavelet_Transform::ApplyTransform: Filter size larger than input image size')
            prop_img = inp_img*psi_filter_bank[i][j]

            #due to symmetry of real signals,
            #it is sufficient to consider only the filters of one half-plane of the frequency domain
            if(i+j<2*num_coronas+1):
                prop_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(prop_img)))
                prop.extend([prop_img])
                current_corona = max(abs(i-num_coronas), abs(j-num_coronas))
                #meta of 'prop_img' contains number of the corona on which 'prop_img' whas generated
                m = {'corona': current_corona}
                new_meta.extend([m])

    return {'prop': prop, 'out': out, 'meta': new_meta}


if __name__ == '__main__':
    """
    Plot gabor filter bank.
    """
    
    import matplotlib.pyplot as plt

    tf = gabortsfm(num_coronas=1, filter_num_rows=64, filter_num_columns=64)

    # Plot filter banks.
    
    psifilters = [val for one_filter_bank in tf._psi_filter_bank for val in one_filter_bank]
    fig, axes = plt.subplots(2*tf._num_coronas+1, 2*tf._num_coronas+1, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.3, wspace=0.05)
    for ax, s in zip(axes.flat, psifilters):
        im = ax.imshow(s, interpolation='nearest')
    plt.show()





