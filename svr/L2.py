import numpy as np
import cvxopt
from svr.kernelfunc import calckernel

def l2svr(xparam,svrinfo,return_all=False):
    """
    Subroutine for calculating L2 LOOCV error of SVR model

    Args:
         xparam (nparray): Array contains hyperparameter values.
         svrinfo (object): Class contains SVR informations.
         return_all (bool): Default to False, only return LOO error value. Otherwise returns all
         values.

    Returns:
        errloo : LOO Error
        mu : mu
    """
    x = svrinfo.x_norm
    y = svrinfo.y

    # Retrieve theta
    try:
        theta = svrinfo.theta
    except AttributeError:
        theta = 10 ** xparam[:svrinfo.nlens]

    # Retrieve epsilon
    try:
        epsilon = svrinfo.epsilon
    except AttributeError:
        epsilon = 10 ** xparam[svrinfo.nlens+svrinfo.nepsi-1]

    # Retrieve C
    try:
        c = svrinfo.c
    except AttributeError:
        c = 10 ** xparam[svrinfo.nlens + svrinfo.nepsi + svrinfo.nc - 1]

    # Retrieve kernel weights
    try:
        wgk = np.array([svrinfo.wgk])
    except AttributeError:
        if svrinfo.nkrnl > 1:
            wgk = 10 ** xparam[svrinfo.nlens + svrinfo.nepsi + svrinfo.nc - 1:]
        else:
            wgk = np.array([1])

    tol = 1e-5
    kerneltype = svrinfo.kerneltype
    covlst = []

    for i in range(svrinfo.nkrnl):
        if svrinfo.lenscale == 'anisotropic':
            cov_i = calckernel(x, x, theta, svrinfo.nvar, ker=kerneltype[i])
        else:
            theta_k = theta * np.ones(svrinfo.nvar)
            cov_i = calckernel(x, x, theta_k, svrinfo.nvar, ker=kerneltype[i])

        covlst.append(cov_i)

    psi = np.zeros((svrinfo.nsamp, svrinfo.nsamp))
    for ii in range(svrinfo.nkrnl):
        psi += (wgk[i]/np.sum(wgk)) * covlst[i]

    # Construct dual variable
    # Matric of corr
    # L2-SVR formulation
    psi += (1/c)*np.eye(svrinfo.nsamp)
    psicon = np.vstack((np.hstack((psi,-psi)),np.hstack((-psi,psi))))

    #constraint
    cn = np.vstack(((epsilon*np.ones((svrinfo.nsamp,1))-y), (epsilon*np.ones((svrinfo.nsamp,1))+y)))

    #lower bound
    lb = np.zeros((2*svrinfo.nsamp,1))

    #upper bound
    ub = c * np.ones((2 * svrinfo.nsamp, 1))

    #optimization constraints
    aeq = np.hstack((-np.ones((1,svrinfo.nsamp)),np.ones((1,svrinfo.nsamp))))
    beq = 0

    alpha_pm = quadprog(psicon, cn, Aeq=aeq, beq=beq, lb=lb, ub=ub)
    alpha = alpha_pm[:svrinfo.nsamp] - alpha_pm[svrinfo.nsamp:]

    # Find support vectors
    sv_i,_ = np.where(abs(alpha)>tol)
    x_sv = x[sv_i,:]
    alpha_sv = abs(alpha[sv_i])
    num_sv = len(x_sv)

    #Find SV mid between 0 and C
    sv_mid_i = np.argmin(abs(abs(alpha)-(c/2)))
    mu = y[sv_mid_i] - epsilon*np.sign(alpha[sv_mid_i]) - np.dot(alpha[sv_i].T, psicon[sv_i,sv_mid_i].reshape(-1,1))

    covksvlst = []

    for i in range(svrinfo.nkrnl):
        if svrinfo.lenscale == 'anisotropic':
            covksv_i = calckernel(x_sv, x_sv, theta, svrinfo.nvar, ker=kerneltype[i])
        else:
            theta_k = theta * np.ones(svrinfo.nvar)
            covksv_i = calckernel(x_sv, x_sv, theta_k, svrinfo.nvar, ker=kerneltype[i])

        covksvlst.append(covksv_i)

    ksv = np.zeros((num_sv, num_sv))
    for ii in range(svrinfo.nkrnl):
        ksv += (wgk[i] / np.sum(wgk)) * covksvlst[i]

    ksv += (1 / c) * np.eye(num_sv)
    ksv_bar = np.vstack((np.hstack((ksv,np.ones((num_sv,1)))),np.hstack((np.ones((1,num_sv)),[[0]]))))

    if np.linalg.cond(ksv_bar) < 1e-8:
        raise ValueError('Matrix is ill-conditioned')

    # Calculate Error LOO
    eta = 0.1
    d = np.diag(eta/alpha_sv.flatten())

    d_bar = np.vstack((np.hstack((d,np.zeros((num_sv,1)))),np.hstack((np.zeros((1,num_sv)),[[0]]))))
    temp1 = 1 / np.diag(np.linalg.pinv(ksv_bar+d_bar))
    temp2 = -np.diag(d_bar)
    sp2 = np.hstack((temp1.reshape(-1,1), temp2.reshape(-1,1)))

    temp_alpha2 = 0
    for i in range(num_sv):
        temp_alpha2 += alpha_sv[i]*sp2[i,0]

    errloo = temp_alpha2/num_sv + epsilon

    if return_all:
        return errloo,mu,alpha,epsilon,theta,c,wgk
    else:
        return (errloo)


def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Subroutine copied from Nolfwin's github https://github.com/nolfwin/cvxopt_quadprog
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if lb is not None and ub is not None:
        if L is None and k is None:
            L = np.vstack([-np.eye(n_var), np.eye(n_var)])
            k = np.vstack([lb, ub])
        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

    return np.array(sol['x'])
