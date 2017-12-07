# Upscaling

Here we focus on the one dimensional case and a set of pairwise symmetric filters. 

## Run the code

`python EnergyDecay.py`

Additional arguments:

1. Cho0se a set of filters (default: simple_highpass). Possible arguments are:
- simple_highpass
- raised_cosine
- wavelet_rect
- stochastic
2. Amount of input signals that shall be scattered (default: 1)
3. Amount of layers, that is, how deep the network should be (default: 7)
4. A directory where the ouputfile shall be stored (by default it creates a folder named outputs at the current directory)

Example input:

`python EnergyDecay.py raised_cosine 2 6 outputs`

Example output:

Filter:  raised_cosine  
Number of inputs:  2  
Number of layers:  6  
Fetching MNIST dataset...  
Resample input images...  
Scattering training data...  
timer started    
Propagation protocol:  
level  0 : - Energy: 1.0  
level  1 : - Energy: 0.785147470486  
level  2 : - Energy: 0.252075419428  
level  3 : - Energy: 0.0402560453871  
level  4 : - Energy: 0.00708757840821  
level  5 : - Energy: 0.00193679235587  
level  6 : - Energy: 0.000550525690997  
Runtime:  1.04121994972229 sec  
Energy stored in:  outputs/raised_cosine_inp=2_lay=6.csv
