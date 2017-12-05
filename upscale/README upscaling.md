## Upscaling

Here we focus on the one dimensional case and a set of pairwise symmetric filters. 

# Run the code

`python EnergyDecay.py`

Additional arguments:

1. Cho0se a set of filters (default: simple_highpass). Possible arguments are:
- simple_highpass
- raised_cosine
- wavelet_rect
2. Amount of input signals that shall be scattered (default: 1)
3. Amount of layers, that is, how deep the network should be (default: 7)
4. A directory where the ouputfile shall be stored (by default it creates a folder named outputs at the current directory)

Example input:

`python EnergyDecay.py raised_cosine 2 6 outputs`

Example output:
