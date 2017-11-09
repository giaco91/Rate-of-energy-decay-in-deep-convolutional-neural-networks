# Rate-of-energy-decay-in-deep-convolutional-neural-networks
Experiments on the energy decay of the propagated signals in deep convolutional neural networks.

The goal of this repo is to serve a python files, that implement a deep convolutaional neural networks with certain sets of well chosen filters. 

## Preinstallations

Make sure you have preinstalled the following:

  - scikit-learn:   http://scikit-learn.org/stable/install.html
  - SciPy:    https://www.scipy.org/install.html

## Run main.py

The user can pass any image of any size, choose, whether it should be interpreted as a 1-d or a 2-d signal, and finally decide how many layers are desired.
  
You can simply run the code e.g. from your terminal:
```$ python main.py ```

By default, an image of the MNIST dataset is used, interpreted as a 2d-signal and scattered over 5 layers of a mixture of filters. The propagated signals go through a rectified linear unit non-linearity and a max-pooling. However, you can pass three arguments as described above:
  1. Path to your image you want to propagate
  
  2. Pass ``` 1 ``` or ``` 2 ``` for an interpretation of the image as a 1d- or 2d-signal.
  
  3. Pass any number from ``` 1 ``` to ``` 20 ``` to choose the amount of layers.
  
Note that the order of the arguments is important and that you can pass either all arguments or none. The used filters are stationary wavelets in 1d and Gabor wavelets in the 2d case.

### Example input:

```$ python main.py image.jpg 1 8```

### Example output: 

preprocessing your signal in 1-dimension...

Propagation protocol:

level  0 : - Energy: 25861707617.3 , Amount of signals: 1 , Signal size: 1048576

level  1 : - Energy: 1280520894.14 , Amount of signals: 1 , Signal size: 524288

level  2 : - Energy: 673823174.618 , Amount of signals: 2 , Signal size: 262144

level  3 : - Energy: 324803150.41 , Amount of signals: 4 , Signal size: 131072

level  4 : - Energy: 152369874.5 , Amount of signals: 8 , Signal size: 65536

level  5 : - Energy: 72621136.4797 , Amount of signals: 16 , Signal size: 32768

level  6 : - Energy: 34961046.2309 , Amount of signals: 32 , Signal size: 16384

level  7 : - Energy: 16437425.9316 , Amount of signals: 64 , Signal size: 8192

level  8 : - Energy: 7385179.14625 , Amount of signals: 128 , Signal size: 4096

## Run 1dDecay.py

Here we focuse on the 1d case. The filters are a normalized dyadic set of highpass filters that completely span the highpass regime. They are supported either only on the positive or on the negative frequencies. We have a modulus non-linearity and no pooling. The input signals are unrolled MNIST digit images of size 32*32=1024. By default only one image is scattered and the highpass filters are two ideal high pass filters, one for the positive and one for the negative frequencies. You can run the file in your terminal: 

```$ python 1dDecay.py ```

There are 3 additional arguments that can be passed: 

1. Integer: How many images do you want to scatter? The output energies at each level will be averaged over the images.
2. Integer {2,6,18}: How many dyadic filters do you want to have at each node? Note that the calculation time scales like: filters^layers. Usually, running out of RAM will be the bigger issue here.
3. Integer: How many layers do you want?

### Example input:

```$ python 1dDecay.py 10 2 11```

### Example output: 

Fetching MNIST dataset...

Resample input images...

Scattering training data...

10  signals have been propagated.

Propagation protocol:

level  0 : - Energy: 1.0 , Amount of signals: 1  Signal size: 1024

level  1 : - Energy: 0.735343092132 , Amount of signals: 1  Signal size: 1024

level  2 : - Energy: 0.213223037069 , Amount of signals: 2  Signal size: 1024

level  3 : - Energy: 0.0397570346769 , Amount of signals: 4  Signal size: 1024

level  4 : - Energy: 0.00906225226072 , Amount of signals: 8  Signal size: 1024

level  5 : - Energy: 0.00215886029644 , Amount of signals: 16  Signal size: 1024

level  6 : - Energy: 0.000543186514969 , Amount of signals: 32  Signal size: 1024

level  7 : - Energy: 0.000150559301693 , Amount of signals: 64  Signal size: 1024

level  8 : - Energy: 4.6091346008e-05 , Amount of signals: 128  Signal size: 1024

level  9 : - Energy: 1.51196190117e-05 , Amount of signals: 256  Signal size: 1024

level  10 : - Energy: 5.19660288051e-06 , Amount of signals: 512  Signal size: 1024

level  11 : - Energy: 1.81654108378e-06 , Amount of signals: 1024  Signal size: 1024



