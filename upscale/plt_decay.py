import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

csv_file=sys.argv[1]
energies = np.genfromtxt(csv_file, delimiter=',')

def get_filter_type(csv_file):
	splits=csv_file.split('/')
	filename=splits[-1]
	splits=filename.split('_')
	return splits

def plt_decay(energies):
	splits=get_filter_type(csv_file)
	filter_type='Simple Highpass'
	if splits[0]=='raised':
		filter_type='Raised Cosines'
	elif splits[0]=='wavelet':
		filter_type='Wavelet Rectangular'
	elif splits[0]=='stochastic':
		filter_type='Stochastic (x1,x2,x3)=('+str(splits[1])+','+str(splits[2])+','+str(splits[3])+')'
	
	num_layers=len(energies)
	t = np.arange(0, num_layers, 1)
	slope, intercept, r_value, p_value, std_err = stats.linregress(t[5:],np.log(energies[5:]))
	plt.xlabel('layer')
	plt.ylabel('log(energy)')
	plt.title('Energy Decay: '+filter_type)
	plt.text(0.5*num_layers, -2, 'slope='+str(slope))
	plt.text(0.5*num_layers, -3.3, 'q='+str(np.exp(-slope)))
	plt.plot(t, np.log(energies), 'b--')
	plt.show()

plt_decay(energies)
