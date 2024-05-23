import numpy as np
from scipy.stats import norm

debug = False
def debug_print(i):
    return debug and i==5

# weighted least squares fit with confidence interval
# i = point to predict
# x = x values
# y = y values
# e = statistical errors on y
# w = tricubic weights
# deg = degree of polynomial (1 or 2)
# alpha = confidence interval
def wls(i, x, y, e, w, deg):
    weights = w/(e**2)
    if debug_print(i): print("e",e)
    if debug_print(i): print("w",w)
    if debug_print(i): print("weights",weights)
    # polyfit convention: coef is sorted with highest power first
    # form A matrix (which corresponds to gradient of parameters for least squares fit) following that convention
    # include weights following weighted least squares convention
    X = np.vander(x, deg+1)
    W = np.diag(weights)
    L = X.dot(np.linalg.inv(X.T.dot(W).dot(X))).dot(X.T.dot(W)) # smoothing matrix
    Y = L.dot(y) # predictions
    y_pred = Y[i]
#    if debug_print(i): np.set_printoptions(threshold=np.inf)
    if debug_print(i): print("L",L)
    if debug_print(i): print("y[i]",y[i])
    if debug_print(i): print("y_pred",y_pred)
    return y_pred, L[i]

# this follows "Semiparametric Regression for the Social Sciences", L. Keele
# which is similar to "A Package of C and Fortran Routines for Fitting Local Regression Models", W. Cleveland, E. Grosse, M.-J. Shyu
# (https://www.netlib.org/a/cloess.pdf)
def ci(y, y_pred, L, alpha):
    R = y-y_pred # residuals
    D = np.trace(L) # residual DOF
    N = len(y) - D
    S = np.sum(R**2)/N # error variance
    V = S*L.dot(L.T) # fit variance
    C = np.sqrt(np.diag(V))
#    if debug: np.set_printoptions(threshold=np.inf)
    if debug: print("L",L)
    if debug: print("R",R)
    if debug: print("D",D)
    if debug: print("S",S)
    if debug: print("V",V)
    if debug: print("C",C)
    cl_factor = norm.ppf((1+alpha)/2)
    if debug: print("cl_factor",cl_factor)
    y_dn = y_pred - cl_factor*C
    y_up = y_pred + cl_factor*C
    # generalized cross validation (for span optimization)
    gcv = S/N
    if debug: print("gcv",gcv)
    return y_dn, y_up, gcv

# span = fraction of points to include in fit
def loess(x, y, e, deg, alpha, span):
    denom = span*(x.max()-x.min())
    tricube = lambda d: np.clip((1 - np.abs(d/denom)**3)**3, 0, 1)
    N = len(x)
    y_pred = np.zeros_like(y)
    L_final = np.diag(np.zeros_like(y))
    for i in range(N):
        w = tricube(np.abs(x[i] - x))
        y_pred[i], L_final[i] = wls(i,x,y,e,w,deg)
#        if debug_print(i): print(y_pred[i], L_final[i].dot(y))

    y_dn, y_up, gcv = ci(y, y_pred, L_final, alpha)
    if debug:
        i = 5
        print("stderr",y_pred[i]-y_dn[i])
        print(y_pred[i], (y_dn[i], y_up[i]))
        import sys
        sys.exit(0)

    return y_pred, (y_dn, y_up), gcv
