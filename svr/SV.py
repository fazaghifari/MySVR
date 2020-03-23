import numpy as np
from svr.L2 import l2svr
from sobolsampling.sobol_new import sobol_points
from svr.kernelfunc import calckernel
from scipy.optimize import minimize


class SVRInfo:
    """
    Converter from dictionary to class.
    """
    def __init__(self, normalize=True, disp=False, **kwargs):
        """
        kwargs:
            Take dictionary as an input: SVRInfo(**yourdictionary)
        """
        checkeddict = self.svrinfocheck(kwargs, normalize, disp)
        self.dict = checkeddict
        for key, value in kwargs.items():
            exec('self.' + key + '= value')

    def svrinfocheck(self, info, normalize, disp):
        """
        Checks SVRInfo
        Args:
            info(dict): Dictionary that contains Kriging information.
            normalize(bool): Normalize samples or not.
            disp(bool): Display text or not
        Return:
            info(dict): Dictionary that contains Kriging information.
        """

        if not all(key in info for key in ('x', 'y')):
            raise AssertionError('key x and y are required.')
        else:
            pass

        assert (np.ndim(info['x']) == 2), "x requires 2 dimensional array with shape = nsamp x nvar"
        assert (np.ndim(info['y']) == 2), "y requires 2 dimensional array with shape = nsamp x 1"

        info['nsamp'], info['nvar'] = np.shape(info['x'])

        if 'lb' not in info:
            info['lb'] = np.min(info['x'], axis=0)
        if 'ub' not in info:
            info['ub'] = np.max(info['x'], axis=0)

        if normalize:
            info['x_norm'] = standardize(info['x'], type='default',
                                         ranges=np.vstack((info["lb"], info["ub"])))
        else:
            info['x_norm'] = info['x']

        # Check and set default value for kernel settings
        if 'kerneltype' not in info:
            if 'lenscale' not in info:
                info['kerneltype'] = ['gaussian']
                info['lenscale'] = 'anisotropic'
                info['nkrnl'] = len(info['kerneltype'])
                if disp:
                    print('Using anisotropic Gaussian kernel as default')
            else:
                info['kerneltype'] = ['gaussian']
                info['nkrnl'] = len(info['kerneltype'])
                if disp:
                    print('Using '+info['lenscale']+' Gaussian kernel as default')
        else:
            if 'lenscale' not in info:
                info['lenscale'] = 'anisotropic'
                info['nkrnl'] = len(info['kerneltype'])
                if disp:
                    print('Using anisotropic kernel as default')
            else:
                info['nkrnl'] = len(info['kerneltype'])

        # Check and set default value for kernel weights
        if 'wgk' not in info:
            if info['nkrnl'] == 1:
                info['nwgk'] = 0
                info['wgk'] = 1
            else:
                info['nwgk'] = info['nkrnl']
        else:
            if info['nkrnl'] == 1:
                info['wgk'] = 1
                if disp:
                    print('Construct SVR using single kernel')
            else:
                if disp:
                    print('Construct SVR using provided kernel weight')

        # Check and set default value for theta settings
        if 'theta' not in info:
            if info['lenscale'] == 'anisotropic':
                info['nlens'] = info['nvar']
            elif info['lenscale'] == 'isotropic':
                info['nlens'] = 1
            else:
                raise AssertionError('Only isotropic and anisotropic lengthscale are available.')
        else:
            info['nlens'] = 1
            if disp:
                print('Using fixed lengthscale value ', info['theta'])

        if 'epsilon' not in info:
            info['nepsi'] = 1  # Use constant epsilon
        else:
            info['nepsi'] = 0
            if disp:
                print('Using fixed epsilon value ', info['epsilon'])

        if 'c' not in info:
            info['nc'] = 1  # Use constant C
        else:
            info['nc'] = 0
            if disp:
                print('Using fixed C value ', info['nc'])

        if 'lbhyp' not in info:
            temp = [[-3]*info['nlens'], [-2]*info['nepsi'], [-1]*info['nc'], [0]*info['nwgk']]
            info['lbhyp'] = np.array([item for sublist in temp for item in sublist])

        if 'ubhyp' not in info:
            temp = [[3]*info['nlens'], [0]*info['nepsi'], [2]*info['nc'], [1]*info['nwgk']]
            info['ubhyp'] = np.array([item for sublist in temp for item in sublist])

        if 'errtype' not in info:
            info['errtype'] = 'L2'
            if disp:
                print('Using default LOO error type, L2')

        if 'nrestart' not in info:
            info['nrestart'] = 1

        if 'optimizer' not in info:
            info['optimizer'] = 'lbfgsb'
            if disp:
                print('Using default optimizer, LBFGSB')

        info['mu'] = None
        info['errloo'] = None
        info['alpha'] = None

        return info


class SVR:
    """
    Support vector regression class for creating an SVR model

    Inputs:
        dictionary (dict): Dictionary that contains necessary information for creating SVR model
        normalize (bool): Normalize the data or not.

    Dictionary details:
    REQUIRED PARAMETERS: These parameters need to be specified manually by the user.
    Otherwise, the process cannot continue.
        - info['x'] : Set of input variables,  2 dimensional array with shape = nsamp x nvar.
        - info['y'] : Set of corresponding system response, 2 dimensional array with shape = nsamp x 1.

    OPTIONAL PARAMETERS: These parameters can be set by the user. If not specified,
    default values will be used (or computed for the experimetntal design and responses)
        - info['ub'] (nparray)      : Input variables upper bound, 1 dimensional array with size nvar
        - info['lb'] (nparray)      : Input variables lower bound, 1 dimensional array with size nvar
        - info['kerneltype'] (list) : Kernel type, list. available kernels: 'gaussian', 'exponential', 'matern32', 'matern52'
            example: info['kerneltype'] = ['gaussian'] or info['kerneltype'] = ['gaussian','matern32']
        - info['lenscale'] (str)    : Lengthscale type, string. 'isotropic' or 'anisotropic'.
        - info['wgk'] (nparray)     : Kernel weighting for multiple kernels, 1 dimensional array with size nkernel.
        - info['epsilon'] (float)   : Epsilon in SVR, it specifies the epsilon tube.
        - info['c'] (float)         : C in SVR, it specifies the regularization parameter.
        - info['theta] (nparray)    : The value of kernel lengthscale.
        - info['nrestart'] (int)    : Number of parameters optimization trials.
        - info['optimizer'] (str)   : Type of optimizer.

    """
    def __init__(self, dictionary, normalize=True):
        self.svrinfo = SVRInfo(normalize=normalize,**dictionary)

    def train(self, disp=True):
        """
        Train SVR model

        Args:
           disp (bool): Display process or not. Default to True.

        Returns:
            None
        """
        if self.svrinfo.nepsi == 0 and self.svrinfo.nc == 0 and self.svrinfo.nlens == 0:
            if disp:
                print("Construct SVR without tuning parameters.")
            self.svrinfo.optimizer = None
            xparamopt = [self.svrinfo.theta, self.svrinfo.epsilon, self.svrinfo.c, self.svrinfo.wgk]

            if self.svrinfo.errtype == 'L2':
                self.svrinfo.errloo, self.svrinfo.mu, self.svrinfo.alpha, self.svrinfo.epsilon, \
                    self.svrinfo.theta, self.svrinfo.c, self.svrinfo.wgk = l2svr(xparamopt, self.svrinfo, return_all=True)
            else:
                raise NotImplementedError('Other options are not yet available')

        else:
            xhyp0_norm = sobol_points(self.svrinfo.nrestart+1, self.svrinfo.nvar)
            xhyp0 = xhyp0_norm[1:,:] * (self.svrinfo.ubhyp - self.svrinfo.lbhyp) + self.svrinfo.lbhyp
            optimbound = np.transpose(np.vstack((self.svrinfo.lbhyp, self.svrinfo.ubhyp)))

            bestxcand = np.zeros(np.shape(xhyp0))
            errloocand = np.zeros(shape=[self.svrinfo.nrestart])
            for ii in range(self.svrinfo.nrestart):
                xhyp0_ii = xhyp0[ii, :]

                if self.svrinfo.optimizer == 'lbfgsb':
                    res = minimize(l2svr, xhyp0_ii, method='L-BFGS-B', options={'eps': 1e-03, 'disp':False},
                                   bounds=optimbound, args=(self.svrinfo, False))
                    bestxcand_ii = res.x
                    errloocand_ii = res.fun
                else:
                    raise NotImplementedError('Other optimizers are not yet implemented')

                bestxcand[ii, :] = bestxcand_ii
                errloocand[ii] = errloocand_ii

            I = np.argmin(errloocand)
            xparamopt = bestxcand[I, :]

            if disp:
                print("Train hyperparam finished.")
                print(f"Best hyperparameter is {xparamopt}")
                print(f"With Error LOO of {errloocand[I]}")

            self.svrinfo.errloo, self.svrinfo.mu, self.svrinfo.alpha, self.svrinfo.epsilon, \
                self.svrinfo.theta, self.svrinfo.c, self.svrinfo.wgk = l2svr(xparamopt, self.svrinfo, return_all=True)

    def predict(self, x):
        theta = self.svrinfo.theta
        kerneltype = self.svrinfo.kerneltype
        covlst = []

        for i in range(self.svrinfo.nkrnl):
            if self.svrinfo.lenscale == 'anisotropic':
                cov_i = calckernel(self.svrinfo.x, x, theta, self.svrinfo.nvar, ker=kerneltype[i])
            else:
                theta_k = theta * np.ones(self.svrinfo.nvar)
                cov_i = calckernel(self.svrinfo.x, x, theta_k, self.svrinfo.nvar, ker=kerneltype[i])

            covlst.append(cov_i)

        psi = np.zeros(np.shape(covlst[0]))
        for ii in range(self.svrinfo.nkrnl):
            psi += (self.svrinfo.wgk[i] / np.sum(self.svrinfo.wgk)) * covlst[i]

        ypred = self.svrinfo.mu + np.dot(self.svrinfo.alpha.T,psi)

        return ypred.T


def standardize(X,y=None,type='default',norm_y=False, ranges = np.array([None])):

    if type.lower()=='default':
        X_norm = np.empty(np.shape(X))
        y_norm = np.empty(np.shape(y))
        if ranges.any() == None:
            raise ValueError("Default normalization requires range value!")
        if norm_y == True:
            # Normalize to [0,1]
            for i in range(0, np.size(X, 1)):
                X_norm[:, i] = (X[:, i] - ranges[0, i]) / (ranges[1, i] - ranges[0, i])
            for jj in range(0, np.size(y, 1)):
                y_norm[:, jj] = (y[:, jj] - ranges[0, 1+i+jj]) / (ranges[1, 1+i+jj] - ranges[0, 1+i+jj])
            # Normalize to [-1,1]
            X_norm = (X_norm - 0.5) * 2
            y_norm = (y_norm - 0.5) * 2

            return X_norm,y_norm
        else:
            #Normalize to [0,1]
            for i in range(0,np.size(X,1)):
                X_norm[:,i] = (X[:,i]-ranges[0,i])/(ranges[1,i]-ranges[0,i])
            #Normalize to [-1,1]
            X_norm = (X_norm-0.5)*2
            return X_norm

    elif type.lower() == 'std':
        if norm_y == True:
            X_mean = np.mean(X, axis=0)
            X_std = X.std(axis=0, ddof=1)
            y_mean = np.mean(y, axis=0)
            y_std = y.std(axis=0, ddof=1)
            X_std[X_std == 0.] = 1.
            y_std[y_std == 0.] = 1.

            X_norm = (X - X_mean) / X_std
            y_norm = (y - y_mean) / y_std
            return X_norm, y_norm, X_mean, y_mean, X_std, y_std
        else:
            X_mean = np.mean(X, axis=0)
            X_std = X.std(axis=0, ddof=1)
            X_std[X_std == 0.] = 1.

            X_norm = (X - X_mean) / X_std
            return X_norm, X_mean, X_std