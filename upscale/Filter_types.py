import numpy as np
import matplotlib.pyplot as plt

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
		#first halve of mirror filters
		if filter_index!=None:
			f=abs(np.fft.ifft(self.apply_filter(np.fft.fft(signal),filter_index)))
			energy=self.get_squared_norm(f)
		#second halve of mirror filters: signals can be calculated from the first
		else:
			f=np.flip(symmetric_signal,0)
			energy=symmetric_energy
		energies=np.zeros(num_layers+1)
		energies[layer]=energy
		#if we are at the deepest layer, we return
		if layer==num_layers:
			return [energies, f, energy]
		#else we recursively call rec with different arguments, depending on whether 
		#the corresponding filter belongs to the first or the second halve of the mirror filter	
		else:
			for k in range(0,num_filters):
				feedback=self.rec(f,layer+1,num_layers,k,num_filters,False,None)
				energies+=feedback[0]+self.rec(None,layer+1,num_layers,None,num_filters,feedback[1],feedback[2])[0]
			return [energies, f, energy]

	def apply_filter(self,fft_signal,filter_index):
		return np.multiply(fft_signal,self.filters[filter_index])

	def plot(self):
		t=np.arange(0, 513, 1)
		plt.xlabel('k')
		plt.ylabel('g_hat[k]')
		plt.title('Filter: '+self.filter_type)
		colors=['b--','r','g-.','c--']
		for i in range(0,self.num_filters):
			plt.plot(t, self.filters[i][0:513], colors[i])
		if self.filter_type=='stochastic':
			plt.text(450, 0.5, 'x1='+str(self.x1))
			plt.text(450, 0.45, 'x2='+str(self.x2))
			plt.text(450, 0.4, 'x3='+str(self.x3))
		plt.show()

#The filters are child classes
class Simple_highpass(Filter_types):

	def __init__(self):
		Filter_types.__init__(self,'simple_highpass')
		self.num_filters=1
		f=np.zeros(1025)
		f[1:513]=1
		self.filters=np.array([f])

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
		f1[513:]=0
		f1[0]=0
		f2=((np.cos(omega*line + np.pi)+1)/2)**(1/2)
		f2[513:]=0
		f2[0]=0
		self.filters=np.array([f1,f2])


class Wavelet_rect(Filter_types):

	def __init__(self):
		Filter_types.__init__(self,'wavelet_rect')
		self.num_filters=4
		z=np.zeros(1025)
		amplitude=1/np.sqrt(2)
		f1=np.copy(z)
		f1[1:26]=amplitude
		f2=np.copy(z)
		f2[6:126]=amplitude
		f3=np.copy(z)
		f3[26:513]=amplitude
		f4=np.copy(z)
		f4[126:513]=amplitude
		self.filters=np.array([f1,f2,f3,f4])


class Stochastic(Filter_types):

	def __init__(self):
		Filter_types.__init__(self,'stochastic')
		self.num_filters=4
		x1=np.random.randint(21,300)
		x2=np.random.randint(20,x1+1)
		x3=np.random.randint(x1,512)	
		self.x1=x1
		self.x2=x2
		self.x3=x3
		print('Filter-bonds: ')
		print('x1:',x1)
		print('x2:',x2)
		print('x3:',x3)
		z=np.zeros(1025)
		amplitude=1/np.sqrt(2)
		f1=np.copy(z)
		f1[20:x1+1]=amplitude
		f2=np.copy(z)
		f2[x2:x3+1]=amplitude
		f3=np.copy(z)
		f3[x1+1:513]=amplitude
		f4=np.copy(z)
		f4[x3+1:513]=amplitude
		self.filters=np.array([f1,f2,f3,f4])





