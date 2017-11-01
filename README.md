# Rate-of-energy-decay-in-deep-convolutional-neural-networks
Experiments on the energy decay of the propagated signals in deep convolutional neural networks.

The goal of this repo is to serve a python file, that implements a deep convolutaional neural network with a certain set of well chosen filters. The user can pass any image of any size, choose, whether it should be interpreted as a 1-d or a 2-d signal, and finally decide how many layers are desired.

## Run main.py

Make sure you have preinstalled the following:

  - scikit-learn:   http://scikit-learn.org/stable/install.html
  - SciPy:    https://www.scipy.org/install.html
  
Then you can simply run the e.g. from your terminal:
```$ python main.py ```

By default, an image of the MNIST dataset is used, interpreted as a 2d-signal and scattered over 10 layers. However, you can pass three arguments as described above:
  1. Path to your image you want to propagate
  
  2. Pass ``` 1 ``` or ``` 2 ``` for an interpretation of the image as a 1d- or 2d-signal.
  
  3. Pass any number from ``` 1 ``` to ``` 20 ``` to choose the amount of layers.
  
Note that the order of the arguments is important and that you can pass either all arguments or none.
Example input:

```$ python main.py image.jpg 1 8```

Example output:

`preprocessing your signal in 1-dimension...

Total propagated energy at:

level  1 :  552907215.611

level  2 :  153676110.639

level  3 :  37964433.8056

level  4 :  8962932.84722

level  5 :  2214675.10764

level  6 :  547426.286458

level  7 :  142753.263021

level  8 :  34595.3559028`


