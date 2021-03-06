# Support Vector Regression Package
MySVR is a Support Vector Regression (SVR) package with multi-kernel feature. Written with a simple style, this package is suitable for anyone who wish to learn about SVR implementation in Python.
 
### Table of contents
- [Example](#example)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [References](#references)

## Example
Example of the package usage can be found in [examples](https://github.com/fazaghifari/MySVR/tree/master/svr/examples) folder. Here an example of 1 dimensional case found in [examples/1d_svr](https://github.com/fazaghifari/MySVR/blob/master/svr/examples/1d_svr.py) is given.

Start by importing all required package:

```python
import numpy as np
from svr.SV import SVR
import matplotlib.pyplot as plt
```

The SVR package requires input variables ***X*** and its corresponding response ***y***, therefore we define the inputs as:

```python
Xsamp = np.array([0,0.5,0.25,0.75,0.125,0.625,0.375,0.875,0.0625,0.5625]).reshape(-1,1)  # The input should be nsamp x nvar
Ysamp = Xsamp * np.sin(Xsamp*np.pi)  # The response should be nsamp x 1
maxY = max(abs(Ysamp))  # For normalizing Y
```

The next step is to define the parameter dictionary details of the dictionary key is available in `help(SVR)`:

```python
svrinfo = dict()
svrinfo['x'] = Xsamp  # Input variables X
svrinfo['y'] = Ysamp/maxY  # Corresponding input response Y (normalized)
svrinfo['epsilon'] = 0.05  # Define the epsilon tube, this parameter is optional 
svrinfo['optimizer'] = 'lbfgsb'  # Define optimizer, this parameter is optional
svrinfo['errtype'] = 'L2'  # Define metric for model training, this parameter is optional
svrinfo['kerneltype'] = ['gaussian','matern52']  # Define kernel type, in this case we use multiple kernel for demo. This parameter is optional 
```

To create and train the model, simply feed the dictionary into the SVR:

```python
model = SVR(svrinfo, normalize=False)
model.train()
```

To predict values, feed your input to the `.predict()` method:

```python
xplot = np.linspace(0,1,100).reshape(-1,1) # Create a set of prediction input
ypred = model.predict(xplot)
```

Finally, plotting:
```python
plt.plot(xplot, ypred * maxY, 'k', label='Prediction')
plt.plot(xplot, (ypred - model.svrinfo.epsilon) * maxY, 'r--', label='Epsilon -')
plt.plot(xplot, (ypred + model.svrinfo.epsilon) * maxY, 'r--', label='Epsilon +')
plt.scatter(Xsamp, Ysamp, c='b', marker='+', label='Samples')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```
The result might looks like the following:
![Example of results](https://github.com/fazaghifari/MySVR/blob/master/images/1dim.png "Results for 1 dimensional SVR prediction.")

## Installation
MySVR works with Python 3, to install write:

```
pip install mysvr
```

## Dependencies

MySVR has the following dependencies:

* `numpy`
* `scipy`
* `matplotlib`
* `sobolsampling`
* `cvxopt`

## Reference
 - Chang, M., & Lin, C. (2005). Leave-One-Out Bounds for Support Vector Regression Model Selection. Neural Computation, 17(5), 1188 1222. https://doi.org/10.1162/0899766053491869
 - UQLab User manual for [Support Vector Regression](https://uqftp.ethz.ch/uqlab_doc_pdf/1.3.0/UserManual_SupportVectorMachinesRegression.pdf)
 - Forrester, A., Sobester, A., & Keane, A. (2008). Engineering Design via Surrogate Modelling: A Practical Guide. John Wiley & Sons.

#### Author
Kemas Zakaria and Ghifari Adam F
