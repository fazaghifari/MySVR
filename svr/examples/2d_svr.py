import sys
sys.path.insert(0, "..")
import numpy as np
from svr.SV import SVR
from svr.errperf import errperf
from sklearn.svm import SVR as sk_SVR
import matplotlib.pyplot as plt
from matplotlib import cm
from sobolsampling.sobol_new import sobol_points

def generate_model():
    nsamp = 40
    nvar = 2
    ub = np.array([10, 15])
    lb = np.array([-5, 0])
    global  x_norm
    x_norm = sobol_points(nsamp, nvar)
    global X
    global y
    X = x_norm * (ub-lb) + lb
    y = branin(X).reshape(-1,1)
    global minY
    global yrange
    maxY = max(y)
    minY = min(y)
    yrange = (maxY - minY)
    ynorm = (y - minY) / yrange

    svrinfo = dict()
    svrinfo['x'] = X
    svrinfo['y'] = ynorm
    # svrinfo['nrestart'] = 5
    svrinfo['epsilon'] = 0.05
    svrinfo['optimizer'] = 'diff_evo'
    svrinfo['errtype'] = 'L2'
    svrinfo['kerneltype'] = ['gaussian']

    model = SVR(svrinfo, normalize=True)
    model.train()

    clf = sk_SVR(epsilon=0.05)
    clf.fit(X,ynorm.flatten())
    params = clf.get_params()

    return model, clf

def predictSVR(model, clf):
    nvar = 2
    neval = 10000

    xx = np.linspace(-5, 10, 100)
    yy = np.linspace(0, 15, 100)
    Xevalx, Xevaly = np.meshgrid(xx, yy)
    Xeval = np.zeros(shape=[neval, 2])
    Xeval[:, 0] = np.reshape(Xevalx, (neval))
    Xeval[:, 1] = np.reshape(Xevaly, (neval))

    yeval = model.predict(Xeval) * yrange + minY
    yeval11 = clf.predict(Xeval) * yrange + minY
    yact = branin(Xeval).reshape(-1,1)

    # Evaluate RMSE
    print("my RMSE = ", errperf(yact,yeval))
    print("sklearn RMSE = ", errperf(yact,yeval11.reshape(-1,1)))
    print("my MARE = ", errperf(yact, yeval, type='mare'))
    print("sklearn MARE = ", errperf(yact, yeval11.reshape(-1,1), type='mare'))

    yeval1 = np.reshape(yeval, (100, 100))
    yeval2 = np.reshape(yeval11, (100, 100))
    x1eval = np.reshape(Xeval[:, 0], (100, 100))
    x2eval = np.reshape(Xeval[:, 1], (100, 100))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1eval, x2eval, yeval1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    samp = ax.scatter(X[:,0],X[:,1],y)

    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    surf1 = ax1.plot_surface(x1eval, x2eval, yeval2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    samp1 = ax1.scatter(X[:, 0], X[:, 1], y)
    plt.show()

def branin(x):
    a = 5.1 / (4 * (np.pi) ** 2)
    b = 5 / np.pi
    c = (1 - (1 / (8 * np.pi)))

    f = (x[:,1] - a * x[:,0] ** 2 + b * x[:,0] - 6) ** 2 + 10 * (c * np.cos(x[:,0]) + 1)
    return f

if __name__ == '__main__':
    svr, clf = generate_model()
    predictSVR(svr, clf)