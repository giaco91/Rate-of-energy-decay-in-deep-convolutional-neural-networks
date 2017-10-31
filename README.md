# Rate-of-energy-decay-in-deep-convolutional-neural-networks
Experiments on the energy decay of the propagated signals in deep convolutional networks.

The goal of this repo is to serve a python file, that implements a deep convolutaional network with a certain set of well chosen filters. The user can pass any any image of any size, choose, whether it should be interpreted as a 1-d or a 2-d signal, and finally decide how many layers are desired.

## Run main.py:

Make sure you have preinstalled the following:

  - scikit-learn:   http://scikit-learn.org/stable/install.html
  - SciPy:    https://www.scipy.org/install.html
  
Then you can simply run the e.g. from your terminal:
```$ main.py ```

By default, an image of the MNIST dataset is used, interpreted as a 2d-signal and scattered over 10 layers. However, you can pass three arguments as described above:
  1. Path to your image you want to propagate
  
  2. Pass ``` 1 ``` or ``` 2 ``` for an interpretation of the image as a 1d- or 2d-signal.
  
  3. Pass any number from ``` 1 ``` to ``` 200 ``` to choose the amount of layers.
  

Example:

```$ main.py image.jpg 1 20```
