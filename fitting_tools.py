import numpy as np
import objective_builders
import optimizations
from objective_builders import normal_CDF

def split_features(leaf):
    """
    Computing the best splits of each feature by minimizing the 
    given objectives within the node rule's limits. 
    
    The output would be the best split of each feature and the resulting
    optimization value (scores for the split)
    """
    
    # dimensionality of data
    d = 1 if leaf.dat.ndim==1 else leaf.dat.shape[0]
    
    scores = np.zeros(d)
    thetas = np.zeros(d)
    
    # objective builder
    objectives = objective_builders.KDE_1D(leaf.dat, leaf.labels, normal_CDF)
    
    # do the 1D optimization (inside the intervals permitted by the rules)
    for i in range(d):
        X = leaf.dat[i,:] if d>1 else leaf.dat
        bracket = (max(leaf.rules[str(i)][0], X.min()),
                   min(leaf.rules[str(i)][1], X.max()))
        thetas[i], scores[i] = optimizations.bisection_min(objectives[i], bracket)
        
    return thetas, scores

def update_rules(new_split, selected_feature, leaf):
    """
    Updating the set of rules of a given leaf node to create
    its left and right children. 
    
    The left child rule will take in the left side of the split, 
    and the right child will take the right side.
    
    NOTE: in case of the right interval (x: theta< x <=b), however, our filter_data
    function always considers '<=' for all types of intervals (which is valid for
    intervals in for the form [a=X.min(), b=X.max()]. But, when a new split theta
    is inroduced we have to consider (a=theta, b=X.max()]. In order to avoid writing
    a new function, we add a very small number to theta, thus [a=theta+epsilon, b=X.max()], 
    which is an approximation for (a=theta, b] for small epsilon values.
    """
    
    eps = 1e-8
    
    # create two copies of the leaf's rules
    left_rules = leaf.rules.copy()
    right_rules = leaf.rules.copy()
    
    # here are the end-points
    (a,b) = leaf.rules[str(selected_feature)]
    
    # rules for left and right children
    if a<new_split<b:
        left_rules[str(selected_feature)] = (a, new_split)
        right_rules[str(selected_feature)] = (new_split+eps, b)
    else:
        raise ValueError('Value of the new split cannot be outside or on the' +
                         'end-points of the current rules.')
        
    return left_rules, right_rules

def filter_data(dat, labels, rules, selected_feature=None):
    """
    Filering the data according to a given set of rules
    """
    if dat.ndim==1:
        (a,b) = rules['0']
        inds_to_del = np.where(np.logical_or(dat<a, b<dat))
        # filtering data and labels
        dat = np.delete(dat, inds_to_del)
        labels = np.delete(labels, inds_to_del)
        
    elif dat.ndim==2:
        # if a feature is given, then only use that for filtering
        if selected_feature:
            (a,b) = rules[str(selected_feature)]
            X = dat[selected_feature,:]
            inds_to_del = np.where(np.logical_or(X<a, b<X))
            # filtering data and labels
            dat = np.delete(dat, inds_to_del, 1)
            labels = np.delete(labels, inds_to_del)
        # if not, look at all the feature one-by-one
        else:
            for i in range(dat.shape[0]):
                (a,b) = rules[str(i)]
                X = dat[i,:]
                inds_to_del = np.where(np.logical_or(X<a, b<X))
                # filtering data and labels
                dat = np.delete(dat, inds_to_del, 1)
                labels = np.delete(labels, inds_to_del)
    
    return dat, labels
