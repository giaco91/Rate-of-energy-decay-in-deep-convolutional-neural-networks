import numpy as np

#super class
class Filter_types():

	def __init__(self, filter_type):
		self.filter_type=filter_type

	def get_squared_norm(self,signal):
		signal=np.absolute(signal)
		signal=signal.flatten().astype(float)
		signal=np.power(signal,2)
		e=np.sum(signal)
		return e

	def rec(self,signal,layer,num_layers,filter_index,num_filters,symmetric_signal,symmetric_energy):
		if filter_index!=None:
			f=abs(np.fft.ifft(self.apply_filter(np.fft.fft(signal),filter_index)))
			energy=self.get_squared_norm(f)
		else:
			f=np.flip(symmetric_signal,0)
			energy=symmetric_energy
		energies=np.zeros(num_layers+1)
		energies[layer]=energy
		if layer==num_layers:
			return [energies, f, energy]
		else:
			for k in range(0,num_filters):
				feedback=self.rec(f,layer+1,num_layers,k,num_filters,False,None)
				energies+=feedback[0]+self.rec(None,layer+1,num_layers,None,num_filters,feedback[1],feedback[2])[0]
			return [energies, f, energy]


#The filters as child classes
class Simple_highpass(Filter_types):

	def __init__(self):
		Filter_types.__init__(self,'simple_highpass')
		self.num_filters=1

	#filteres in frequency domain
	def apply_filter(self,fft_signal,filter_index):
		fft_signal[0]=0
		fft_signal[513:1025]=0
		return fft_signal

class Raised_cosine(Filter_types):

	def __init__(self):
		Filter_types.__init__(self,'raised_cosine')
		self.num_filters=2
		omega=np.pi/1025	 
		line=np.linspace(0,1024,num=1025)
		f1=((np.cos(omega*line)+1)/2)**(1/2)
		f1[0:513]=0
		f2=((np.cos(omega*line + np.pi)+1)/2)**(1/2)
		f2[0:513]=0
		self.filter1=f1
		self.filter2=f2

	def apply_filter(self,fft_signal,filter_index):	
		if filter_index==0:
		    return np.multiply(fft_signal,self.filter1)
		else:
		    return np.multiply(fft_signal,self.filter2)

class Wavelet_rect(Filter_types):

	def __init__(self):
		Filter_types.__init__(self,'wavelet_rect')
		self.num_filters=4
		z=np.zeros(1025)
		f1=np.copy(z)
		f1[1:26]=1/np.sqrt(2)
		f2=np.copy(z)
		f2[6:126]=1/np.sqrt(2)
		f3=np.copy(z)
		f3[26:513]=1/np.sqrt(2)
		f4=np.copy(z)
		f4[126:513]=1/np.sqrt(2)
		self.filter1=f1
		self.filter2=f2
		self.filter3=f3
		self.filter4=f4

	def apply_filter(self,fft_signal,filter_index):
		#r=5
		if filter_index==0:
			return np.multiply(fft_signal,self.filter1)
		elif filter_index==1:
			return np.multiply(fft_signal,self.filter2)			
		elif filter_index==2:
			return np.multiply(fft_signal,self.filter3)			
		else:
			return np.multiply(fft_signal,self.filter4)			
		






