import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

csv_file=sys.argv[1]
energies = np.genfromtxt(csv_file, delimiter=',')


def plt_decay(energies):
	num_layers=len(energies)
	t = np.arange(0, num_layers, 1)
	slope, intercept, r_value, p_value, std_err = stats.linregress(t,np.log(energies))
	plt.xlabel('layer')
	plt.ylabel('log(energy)')
	plt.title('Energy Decay over the layers')
	plt.text(0.5*num_layers, -1, 'slope='+str(slope))
	plt.plot(t, np.log(energies), 'r--')
	plt.show()

plt_decay(energies)