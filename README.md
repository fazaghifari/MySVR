# Support Vector Regression Package
MySVR is a Support Vector Regression (SVR) package with multi-kernel feature. Written with a simple style, this package is suitable for anyone who wished to learn about SVR implementation in Python.
 
### Table of contents
- [Example](#example)
- [Dependencies](#dependencies)

## Example
Example of the package usage can be found in [examples](https://github.com/fazaghifari/MySVR/tree/master/svr/examples) folder. Here an example of 1 dimensional case found in [examples/1d_svr](https://github.com/fazaghifari/MySVR/blob/master/svr/examples/1d_svr.py) is given.

Start by importing all required package:

```python
import numpy as np
from svr.SV import SVR
import matplotlib.pyplot as plt
```

The SVR package requires input variables $X$ and its corresponding response $y$


### Dependencies

MySVR has the following dependencies:

* `numpy`
* `scipy`
* `matplotlib`
* `sobolsampling`
* `cvxopt`
