
import numpy as np
import inspect
from itertools import zip_longest

class scattering_tree:
    
    def __init__(self, propagation_filters, nonlinearities, poolings):
        """Initialize the scattering transform.
        
        Parameters
        ----------
        propagation_filters :   List of propagation filters.
        nonlinearities      :   List of non-linearities. One per tree layer is required.
                                `None` can be used as `f(x) = x`.
        poolings            :   List of pooling functions. One pooling function per tree layer is required.
                                `None` can be used to disable pooling.
        """
        
        self._prop_filters = propagation_filters
        self._nonlin = nonlinearities
        self._pools = poolings
        
    def scatter(self, signal):
        """Applies the scattering transform onto the signal.
        """
        
        return _scatter(signal, self._prop_filters, self._nonlin, self._pools)
        
    def scatter_flat(self, signal):
        """
        Applies the scattering transform onto the signal and returns the result as a flat numpy array of floats.
        Complex values get splittet into their real and imaginary part.
        """
        
        #return _flatten(self.scatter(signal))

        #--we dont need the ouput
        return self.scatter(signal)
        
    def transform(self, X):
        """Apply the scattering transform on X.
        
        This is a wrapper function for `scatter()` for conformance with the sklearn API.
        """
        return self.scatter_flat(X)
        
    def fit(self, X):
        """Dummy function for sklearn conformance.
        """
        pass


def _flatten(scattered):
    """Convert multi dimensional signals into a single one dimensional vector.
    Complex values get splittet into their real and imaginary part.
    """
 
    flat = [s.flatten() for s in scattered]
    flat = _split_complex(flat)
    if flat==[]:
        print('Warning: It seems no outputs are generated')
        return flat
    else:
        return np.concatenate(flat)

def _scatter(signal, propagation_filters, nonlinearities, poolings):
    """
    Parameters
    ----------
    
    signal  :   Either a numpy array or a list of numpy arrays.
    """
    
    output = []
    nodes = []
    
    # This should allow for multiple input signals (as a list) such as R,G,B channels of an image.
    if type(signal) is list:
        nodes.extend(signal)
    else:
        nodes.append(signal)
    meta = [None]*len(nodes)
    #scatter the tree up to second last layer
    #len(propagation_filters) must equal the amount of levels
    energies=np.zeros(len(propagation_filters)+1)
    energies[0]=get_squared_norm(signal)
    amount_of_props=np.zeros(len(propagation_filters)+1)
    amount_of_props[0]=1
    size_of_props=np.zeros(len(propagation_filters)+1)
    size_of_props[0]=len(signal)

    i=0
    for prop_filters, nl, pooling in zip_longest(propagation_filters, nonlinearities, poolings):
        next_nodes = []
        next_meta = []
        # scatter each node on that level
        e=0
        amount=0
        size=0
        for n, m in zip(nodes, meta):
            sc = scatter_single_node(n, prop_filters, nl, pooling, meta=m)
            #sc is a dictionary with keys: prop,out,meta
            #in prop there are all scattered signals of the node
            output.extend(sc['out'])
            next_nodes.extend(sc['prop'])#das sind die propagierten signale
            e+=get_energy(sc['prop'])
            amount+=1
            size=len(sc['prop'][0])
            next_meta.extend(sc['meta'])
        energies[i+1]=e
        amount_of_props[i+1]=amount
        size_of_props[i+1]=size
        nodes = next_nodes
        meta = next_meta
        i+=1   
    #print_energy(energies,amount_of_props,size_of_props)
    output = _split_complex(output)
    return {'output': output, 'energies': energies, 'amount': amount_of_props, 'size': size_of_props}
    #return output

def _split_complex(arr):
    """Split complex arrays into their real and imaginary part.
    
    If an output image is complex, it is replaced by its real part, and its imaginary part gets
    added as seperate output image at the end of list of output images.
    """
    
    real = []
    
    for x in arr:
        if np.iscomplexobj(x):
            """
            Split complex arrays into their real and imaginary part.
            """
            real.append(np.real(x))
            real.append(np.imag(x))
        else:
            real.append(x)
    
    return real

def scatter_single_node(signal, propagation_filters, nonlinearity, pooling, meta=None):
    """Do a single level scattering of the input image.
    
    Parameters
    ----------
    
    signal              :	Input image.
    filters             :	List of linear filter functions. Can be None.
    nonlinearity        :	Function mapping reals onto reals. Can be None.
    pooling             :  	Pooling function for dimension reduction. Can be None.
    
    meta : Optional meta information.
    
    Returns a dict containing values to be propagated and values for output.
        {'prop': [...], 'out': [...]}
    """
    in_shape = signal.shape
    
    if propagation_filters is None:
        propagation_filters = []
    
    
    # Space for filter outputs.
    filtered = []
    output = []
    new_meta = []
    
    """
    Apply each filter and nonlinearity.
    """
    for f in propagation_filters:
        if f is None:
            filtered.append(signal)
        else:
            
            filter_function = f.apply_filter
            
            # get arguments of f.apply_filter
            f_args = inspect.getargspec(filter_function).args
            
            if len(f_args) > 2:
                # Pass meta data only to filters that want it.
                res = filter_function(signal, meta=meta)
            else:
                # Be nice to filters that can't handle meta information.
                res = filter_function(signal)

            if type(res) is dict:
                #res is a dictionary with prop,out,meta as keys
                filtered.extend(res['prop'])
                if 'out' in res:
                    output.extend(res['out'])
                """
                Store meta data if filter provides it.
                """
                if 'meta' in res:
                    new_meta.extend(res['meta']);
                else:
                    new_meta.extend([None]*len(res['prop']))
            else:
                #res is not a dict
                """
                This filter returned an ordinary array.
                """
                filtered.extend(res)

    
    # Something has gone wrong if this does not hold:
    assert(len(filtered) == len(new_meta));

    
    
    if nonlinearity is not None:
        filtered = [nonlinearity(x) for x in filtered]
    if pooling is None:
        pooled = filtered
    else:
        pooled = [pooling.apply_pooling(x) for x in filtered]   
    return {'prop': pooled, 'out': output, 'meta': new_meta}

def get_energy(signals):
    #must be a list of arrays, 
    #eg. the prop output of the function: scatter_single_node
    e=0
    if type(signals)==list:
        for f in signals: 
            e+=get_squared_norm(f)
    else:
        raise ValueError('The input of get_energy must be a list!')
    return e

def get_squared_norm(signal):
    #must be an array of any dimension
    signal=np.absolute(signal)
    signal=signal.flatten().astype(float)
    signal=np.power(signal,2)
    e=np.sum(signal)
    return e 

def print_energy(e, amount,size):
    i=0
    print('Propagation protocol:')
    for e in e:
        print('level ',i,': - Energy:', e, ', Amount of signals:', int(amount[i]), ' Signal size:', int(size[i]))
        i+=1






