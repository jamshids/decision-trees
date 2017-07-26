import numpy as np
import copy
import pdb
from sklearn import tree
import gen_data
import fitting_tools
import tree_structure
import objective_builders

def eval_posteriors(X, T, lambdas):
    """Function for computing the posterior probability of some
    test samples using a given trained tree
    
    The output of this function includes two types of posterior
    probabilities. One is the empirical estimation, and one is
    based on KDE.
    
    lambdas are class priors to be used for Laplace correction
    """
    
    # total number of samples
    n = len(X) if X.ndim==1 else X.shape[1]
    c = len(T.symbols)
    
    # first, compute the KDE-based  posteriors at once
    KDE_posts_mat,_ = T.posteriors_predict(X)
    
    # Second, compute the empirical posteriors, one-by-one
    emp_posts_mat =  np.zeros((c,n))
    Lap_posts_mat =  np.zeros((c,n))
    for i in range(n):
        x = X[i] if X.ndim==1 else X[:,i]
        # extracting the leaf which x belongs to
        leaf_ind = T.extract_leaf(x)
        for j in range(c):
            leaf = T.node_dict[str(leaf_ind)]
            class_count = np.sum(leaf.labels==T.symbols[j])
            total_count = len(leaf.labels)
            # empirical
            emp_posts_mat[j,i] = class_count/float(total_count)
            # after laplace correction
            Lap_posts_mat[j,i] = (class_count+lambdas[j]) / \
                (float(total_count)+1)
            
    return KDE_posts_mat, emp_posts_mat, Lap_posts_mat


def compare_posterior_estimation(n_list):
    """Generating synthetic GMM data, estimating class probabilities
    based on decision trees with empirical and KDE-based estimations
    with a single size of the data set

    Evaluation is done for different number of training data points.
    The input argument should be a list of training data sizes where
    each member of the list is a 3-element tuple indicating size of
    each class of the training set.
    """
    
    """generating fixed-szie test data set"""
    X_test, Y_test, specs_list = gen_data.generate_class_GMM(500, 500, 1000)
    
    """Evaluate posterior estimation based on trees trained with training
    data set of different sizes"""
    KDE_MSE = np.zeros(len(n_list))
    emp_MSE = np.zeros(len(n_list))
    Lap_MSE = np.zeros(len(n_list))
    
    for i in range(len(n_list)):
        n1, n2, n3 = n_list[i]
        X_train, Y_train, _ = gen_data.generate_class_GMM(n1, n2, n3)
    
        # training original CART tree
        sklearn_T = tree.DecisionTreeClassifier()
        sklearn_T.fit(np.transpose(X_train), Y_train)
        # converting it to KDE-based format
        T = tree_structure.convert_SK(sklearn_T, X_train, Y_train, 
                                      objective_builders.normal_CDF)
        # training the tree based on the KDE-based training
        #KDE_T = tree_structure.Tree(X_train, Y_train, objective_builders.normal_CDF)
        #KDE_T.fit_full_tree()    
        
        # true posteriors
        posteriors = np.zeros((3, X_test.shape[1]))
        posteriors[0,:] = gen_data.eval_GMM(X_test, specs_list[0]) * 200. / 800.
        posteriors[1,:] = gen_data.eval_GMM(X_test, specs_list[1]) * 200. / 800.
        posteriors[2,:] = gen_data.eval_GMM(X_test, specs_list[2]) * 200. / 800.
        post_sums = np.repeat(np.expand_dims(np.sum(posteriors,axis=0),axis=0), 3, axis=0)
        posteriors /= post_sums
        # estimated posteriors
        lambdas = np.array([n1, n2, n3]) / float(n1+n2+n3)
        est_posteriors = eval_posteriors(X_test, T, lambdas)
        
        # compute MSE loss of the posterior estimations
        KDE_MSE[i] = np.sum(np.sum((posteriors - est_posteriors[0])**2, axis=0)/3.) / float(n1+n2+n3)
        emp_MSE[i] = np.sum(np.sum((posteriors - est_posteriors[1])**2, axis=0)/3.) / float(n1+n2+n3)
        Lap_MSE[i] = np.sum(np.sum((posteriors - est_posteriors[2])**2, axis=0)/3.) / float(n1+n2+n3)
        
        
    return KDE_MSE, emp_MSE, Lap_MSE
    
    

    
    
