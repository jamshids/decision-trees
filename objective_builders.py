import numpy as np
from scipy.stats import norm

def KDE_1D(dat, labels, kernel_CDF, class_priors=None):
    """
    Constructing the objective function evaluator for an isotropic KDE. 
    
    We need training data samples as well as kernel CDF for computing the objective
    function. The, kernel CDF is a single callable function with a set of parameters. Each dimension
    (feature component) possibly has a different set of parameters. The argument |sigma| is a list of
    parameter sets corresponding to all the dimensions (hence, its length should be equal to the
    number of features in |dat|). Each time, |kernel_CDF| has to be called using a particular set of
    parameters, and a specific data point.
    
    CAUTIOUN 1: |kernel_CDF| should have three arguments only; 
        
        * the first one is the set of parameters:math:`\\Lambda`, 
        * the second one is the sample point on which the kernel is centered, :math:`x_i`, 
        * the last is the value at which the CDF is to be evaluated :math`\\theta`. 
    
    CAUTIOUN 2: |kernel_CDF| should be able to evaluate CDF of the kernel at several data samples in 
    a single call.
    """
    
     
    if dat.ndim==1:
        d = 1
        n = len(dat)
    else:
        d, n = dat.shape
    
    # first of all, we need the priors:
    if not(class_priors):
        symbols = np.unique(labels)
        c = len(symbols)
        class_priors = np.zeros(c)
        for i in range(c):
            class_priors[i] = np.sum(labels==symbols[i]) / float(n)
   
    pies = class_priors
    
    # list of objectives for each dimension
    objectives = [0]*d     # (list initialization)
    
    for i in range(d):
        
        X = dat[i,:] if d>1 else dat
        uncond_sigma = .5
        cond_sigma = .5
        
        # constructing CDF of the unconditional KDE
        marginal_CDF = lambda theta: kernel_CDF(uncond_sigma, X, theta)
        
        # constructing CDF of class-conditional KDE
        class_marginals = [0]*c   # (list initialization)
        for j in range(c):
            class_marginals[j] = lambda theta: kernel_CDF(cond_sigma, X[labels==symbols[j]], theta)
            
        # constructing the overall objective of this i-th feature:
        J_i = lambda theta: np.sum(pies*np.array([f(theta)*np.log(marginal_CDF(theta)/f(theta)) + 
                                                  (1-f(theta))*np.log((1-marginal_CDF(theta))/(1-f(theta))) 
                                                  for f in class_marginals]))
        objectives[i] = J_i
        
    return objectives

def normal_CDF(sigma, dat, theta):
    """
    CDF of a scalar KDE based on Gaussian kernels. 
    
    |sigma| is a positive scalar denoting the variance of the Gaussian kernel.
    """
    return np.sum(norm.cdf((theta - dat)/sigma)) / len(dat)
    
