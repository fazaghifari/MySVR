import sys
sys.path.insert(0, "..")
import numpy as np
from svr.SV import SVR
import matplotlib.pyplot as plt

Xsamp = np.array([0,0.5,0.25,0.75,0.125,0.625,0.375,0.875,0.0625,0.5625]).reshape(-1,1)
Ysamp = Xsamp * np.sin(Xsamp*np.pi)
maxY = max(Ysamp)
minY = min(Ysamp)
yrange = (maxY - minY)
ynorm = (Ysamp - minY) / yrange

svrinfo = dict()
svrinfo['x'] = Xsamp
svrinfo['y'] = ynorm
svrinfo['lb'] = np.array([0])
svrinfo['ub'] = np.array([1])
svrinfo['epsilon'] = 0.05
svrinfo['optimizer'] = 'lbfgsb'
svrinfo['errtype'] = 'L2'
svrinfo['kerneltype'] = ['gaussian','matern52']

model = SVR(svrinfo, normalize=True)
model.train()

xplot = np.linspace(0,1,100).reshape(-1,1)
ypred = model.predict(xplot)

plt.plot(xplot, ypred * yrange + minY, 'k', label='Prediction')
plt.plot(xplot, (ypred - model.svrinfo.epsilon) * yrange + minY, 'r--', label='Epsilon -')
plt.plot(xplot, (ypred + model.svrinfo.epsilon) * yrange + minY, 'r--', label='Epsilon +')
plt.scatter(Xsamp, Ysamp, c='b', marker='+', label='Samples')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
