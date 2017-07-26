import numpy as np
from numpy import matrix as mat
import copy
import pdb
import matplotlib.pyplot as plt


def circle_dat(n1,n2,n3, flag=False):

    r1 = 1
    r2 = 2
    r3 = 3

    angles = 2*np.pi*np.random.random(n1)
    r = r1 + .2*np.random.randn(n1)
    X11 = r * np.cos(angles - np.pi)
    X12 = r * np.sin(angles)

    angles = 2*np.pi*np.random.random(n1)
    r = r2 + .2*np.random.randn(n1)
    X21 = r * np.cos(angles - np.pi)
    X22 = r * np.sin(angles)

    angles = 2*np.pi*np.random.random(n1)
    r = r3 + .2*np.random.randn(n1)
    X31 = r * np.cos(angles - np.pi)
    X32 = r * np.sin(angles)
    
    if flag:
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        
        plt.plot(X11, X12, '*', color='b')
        plt.plot(X21, X22, '*', color='r')
        plt.plot(X31, X32, '*', color='g')
        plt.axis('equal')
        
        ax.set_xticklabels([])
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        ax.set_yticklabels([])
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False


    X_1 = np.array([X11,X12])
    X_2 = np.array([X21,X22])
    X_3 = np.array([X31,X32])
    X_total = np.concatenate((X_1, X_2, X_3), axis=1)
    Y_total = np.ones(n1+n2+n3)
    Y_total[n1:n1+n2] = 2.
    Y_total[n1+n2:] = 3.
    
    return X_total, Y_total


def generate_GMM(specs, n):
    """Generating samples from a GMM
    
    The inputs argument is a list of tuples, where each tuple 
    includes the specificatins of each Gaussian component.
    """
    
    # number of components
    n_comp = len(specs)
    
    # setting the cumulative of priors 
    pies = np.zeros(n_comp)
    for i in range(n_comp):
        pies[i] = specs[i][0]
    cumpies = np.cumsum(pies)
    
    # allocating a variable to contain the samples
    d = 1 if specs[0][1].ndim<1 else len(specs[0][1])
    samples = np.zeros((d, n))
    for i in range(n):
        # random selection of the Gaussian component
        u = np.random.random()
        sel_comp = np.where(u <= cumpies)[0][0]
        # generating sample from that component
        mean, cov = specs[sel_comp][1], specs[sel_comp][2]
        samples[:,i] = np.random.multivariate_normal(mean, cov)
        
    return samples

def generate_class_GMM(n1, n2, n3, plot_flag=False):
    """Generate samples from several GMM  classes with 
    fixed parameters
    """
    
    # first class
    mean11 = np.array([0,0.])
    cov11 = np.array([[2.,0.], [0,1]])
    mean12 = np.array([-2.,4.])
    cov12 = np.array([[.1,.0], [0,5.]])
    mean13 = np.array([1.,8.])
    cov13 = np.array([[2.5,0.], [0.,1.5]])
    pies1 = [.3, .3, .4]
    specs1 = [(pies1[0], mean11, cov11), 
             (pies1[1], mean12, cov12),
             (pies1[2], mean13, cov13)]
    X1 = generate_GMM(specs1, n1)

    # second class
    mean21 = np.array([1.,4.])
    cov21 = np.array([[1.5,0.], [0,1]])
    mean22 = np.array([3.,0.])
    cov22 = np.array([[.1,.0], [0,5.]])
    mean23 = np.array([1.,-4.])
    cov23 = np.array([[2.5,0.], [0.,1]])
    pies2 = [.3, .3, .4]
    specs2 = [(pies2[0], mean21, cov21), 
             (pies2[1], mean22, cov22),
             (pies2[2], mean23, cov23)]
    X2 = generate_GMM(specs2, n2)

    # third class
    mean31 = np.array([1.,-8.])
    cov31 = np.array([[3.5,0.], [0,1]])
    mean32 = np.array([-3.,2.])
    cov32 = np.array([[.1,.0], [0,20.]])
    mean33 = np.array([1., 12.])
    cov33 = np.array([[3.5,0.], [0.,1]])
    mean34 = np.array([4.5, 2.])
    cov34 = np.array([[.1,0.], [0.,20.]])
    pies3 = [.25, .25, .25, .25]
    specs3 = [(pies3[0], mean31, cov31), 
             (pies3[1], mean32, cov32),
             (pies3[2], mean33, cov33),
             (pies3[3], mean34, cov34)]
    X3 = generate_GMM(specs3, n3)
    
    if plot_flag:
        plt.scatter(X1[0,:], X1[1,:], color='c')
        plt.scatter(X2[0,:], X2[1,:], color='r')
        plt.scatter(X3[0,:], X3[1,:], color='g')
        
    X = np.concatenate((X1, X2, X3), axis=1)
    
    # also give out the labels
    Y = np.ones(n1+n2+n3)
    Y[n1:n1+n2] = 2.
    Y[n1+n2:] = 3.
    
    return X, Y, [specs1, specs2, specs3]

def eval_GMM(X, specs):
    """Computing likelhiood value of a given GMM
    
    The arguments include test samples and speciications of the
    GMM's components.
    """
    
    # number of components
    n_comp = len(specs)
    
    # sample size and dimensionality
    # (input sample cannot be a scalar)
    n = len(X) if X.ndim<1 else X.shape[1]
    d = 1 if X.ndim<1 else X.shape[0]
    
    # initializing the posteriors to zero
    likelihoods = np.zeros(n)
    for i in range(n_comp):
        pie, mean, cov = specs[i]
        # likelihood of this component
        if d==1:
            norms = ((X - mean)**2) / cov
        else:
            re_mean = X - np.repeat(np.expand_dims(mean, axis=1), n ,axis=1)
            inv_cov = np.linalg.inv(cov)
            dists = np.diag(np.linalg.multi_dot((re_mean.T, inv_cov, re_mean)))
            
        likelihoods += pie * np.exp(-.5*dists)/(2*np.pi*np.linalg.det(cov))
        
    return likelihoods
