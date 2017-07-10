import numpy as np
from objective_builders import KDE_entropy as OBJ_EVAL

def bisection_min(J_theta, bracket):
    
    # end-points
    a,b = bracket
    J_a, J_b = (J_theta(a), J_theta(b))
    
    # central mid-points
    x_m = (a+b) / 2.
    J_m = J_theta(x_m)    
    
    tol = 1e-6
    
    # repeat bisectioning until the length of the bracket is small enough
    while(np.abs(a-b) > tol):

        # left and right mid-points
        x_l = (a + x_m) / 2.
        x_r = (b + x_m) / 2.
        J_r, J_l = (J_theta(x_r), J_theta(x_l))
        
        J_min = np.min([J_a, J_b, J_m, J_r, J_l])
        
        # updating
        if (J_min == J_a) | (J_min == J_l):
            b, J_b = (x_m, J_m)            
            x_m, J_m = (x_l, J_l)
        elif J_min == J_m:
            a,b = (x_l, x_r)
            J_a, J_b = (J_l, J_r)
        elif (J_min == J_b) | (J_min == J_r):
            a, J_a = (x_m, J_m)            
            x_m, J_m = (x_r, J_r)
        
    return x_m, J_m

def minimize_KDE_entropy(dat, labels, kernel_CDF, uncond_sigma=None,
                         cond_sigma=None, priors=None):
    """Function for running bisection-method to find a local minimum
    of the entropy-based impurity objective
    """
    
    # the initial bracket is chosen to be minimum and maximum value of
    # the given scalar data
    (a, b) = (dat.min(), dat.max())
    (J_a, J_b) = (OBJ_EVAL(dat, labels, a, kernel_CDF, uncond_sigma, cond_sigma), 
                  OBJ_EVAL(dat, labels, b, kernel_CDF, uncond_sigma, cond_sigma))
    
    # central mid-points
    x_m = (a+b) / 2.
    J_m = OBJ_EVAL(dat, labels, x_m, kernel_CDF, uncond_sigma, cond_sigma)
    
    tol = 1e-6
    
    # repeat bisectioning until the length of the bracket is small enough
    while(np.abs(a-b) > tol):

        # left and right mid-points
        x_l = (a + x_m) / 2.
        x_r = (b + x_m) / 2.
        J_r, J_l = (OBJ_EVAL(dat, labels, x_r, kernel_CDF, uncond_sigma, cond_sigma), 
                    OBJ_EVAL(dat, labels, x_l, kernel_CDF, uncond_sigma, cond_sigma))
        
        J_min = np.min([J_a, J_b, J_m, J_r, J_l])
        
        # updating
        if (J_min == J_a) | (J_min == J_l):
            b, J_b = (x_m, J_m)            
            x_m, J_m = (x_l, J_l)
        elif J_min == J_m:
            a,b = (x_l, x_r)
            J_a, J_b = (J_l, J_r)
        elif (J_min == J_b) | (J_min == J_r):
            a, J_a = (x_m, J_m)            
            x_m, J_m = (x_r, J_r)
            
    return x_m, J_m
    
    
