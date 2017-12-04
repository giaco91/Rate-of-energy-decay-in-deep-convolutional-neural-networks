import numpy as np

#super class
class Filter_types():

	def __init__(self, filter_type):
		self.filter_type=filter_type

	def get_squared_norm(self,signal):
		signal=np.absolute(signal)
		signal=signal.flatten().astype(float)
		#signal=np.absolute(signal)
		signal=np.power(signal,2)
		e=np.sum(signal)
		return e

	def rec(self,signal,layer,num_layers,filter_index,num_filters):
		#filtering with modulus pooling
		f=abs(np.fft.ifft(self.apply_filter(np.fft.fft(signal),filter_index)))
		energies=np.zeros(num_layers+1)
		if layer==num_layers:
			energies[layer]=self.get_squared_norm(f)
			return energies
		else:
			for k in range(0,num_filters):
				energies+=self.rec(f,layer+1,num_layers,k,num_filters)
			energies[layer]+=self.get_squared_norm(f)
			return energies


#The filters as child classes
class Simple_highpass(Filter_types):

	def __init__(self):
		Filter_types.__init__(self,'simple_highpass')
		self.num_filters=2

	# def rec(self,signal,layer,num_layers,filter_index,num_filters):
	# 	return super(Simple_highpass,self).rec(signal,layer,num_layers,filter_index,self.num_filters)

	#filteres in frequency domain
	def apply_filter(self,fft_signal,filter_index):
	    z=np.zeros(1024)
	    if filter_index == 0:
	        z[1:512]=1
	        return np.multiply(fft_signal,z)
	    else:
	        z[512:1023]=1
	        return np.multiply(fft_signal,z)

class Raised_cosine(Filter_types):

	def __init__(self):
		Filter_types.__init__(self,'raised_cosine')
		self.num_filters=4

	def apply_filter(self,fft_signal,filter_index):
		
		omega=np.pi/1024	 
		line=np.linspace(0,1023,num=1024)
		
		if filter_index==0:
		    rc=((np.cos(omega*line)+1)/2)**(1/2)
		    rc[0]=0
		    rc[512:1024]=0
		    return np.multiply(fft_signal,rc)
		if filter_index==1:
		    rc_shifted=((np.cos(omega*line + np.pi)+1)/2)**(1/2)
		    rc_shifted[0]=0
		    rc_shifted[512:1024]=0
		    return np.multiply(fft_signal,rc_shifted)
		if filter_index==2:
		    rc=((np.cos(omega*line)+1)/2)**(1/2)
		    rc[1023]=0
		    rc[0:512]=0
		    return np.multiply(fft_signal,rc)
		else:
		    rc_shifted=((np.cos(omega*line + np.pi)+1)/2)**(1/2)
		    rc_shifted[1023]=0
		    rc_shifted[0:512]=0
		    return np.multiply(fft_signal,rc_shifted)


