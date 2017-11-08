# Rate-of-energy-decay-in-deep-convolutional-neural-networks
Experiments on the energy decay of the propagated signals in deep convolutional neural networks.

The goal of this repo is to serve a python file, that implements a deep convolutaional neural network with a certain set of well chosen filters. 

## Preinstallations

Make sure you have preinstalled the following:

  - scikit-learn:   http://scikit-learn.org/stable/install.html
  - SciPy:    https://www.scipy.org/install.html

## Run main.py

The user can pass any image of any size, choose, whether it should be interpreted as a 1-d or a 2-d signal, and finally decide how many layers are desired.
  
You can simply run the code e.g. from your terminal:
```$ python main.py ```

By default, an image of the MNIST dataset is used, interpreted as a 2d-signal and scattered over 5 layers of a mixture of filters. However, you can pass three arguments as described above:
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

Here we focuse on the 1d case. The filters are a normalized dyadic set of highpass filters that completely span the highpass regime. They are supported either only in the positive or the negative frequency regime. The input signals are unrolled MNIST digit images of size 32*32=1024. 



