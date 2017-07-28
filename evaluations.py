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
    Smyth_posts_mat = np.zeros((c,n))
    for i in range(n):
        x = X[i] if X.ndim==1 else X[:,i]
        # extracting the leaf which x belongs to
        leaf_ind, path = T.extract_leaf(x)
        # attributes included in the path
        atts = []
        for t in range(len(path)-1):
            atts += [int(T.node_dict[str(path[t])].rule.keys()[0])]
        atts = np.unique(atts)
        
        for j in range(c):
            leaf = T.node_dict[str(leaf_ind)]
            class_count = np.sum(leaf.labels==T.symbols[j])
            total_count = len(leaf.labels)
            # empirical
            emp_posts_mat[j,i] = class_count/float(total_count)
            # after laplace correction
            Lap_posts_mat[j,i] = (class_count+lambdas[j]) / \
                (float(total_count)+1)
            # Smyth's KDE-based formulation
            Xleaf = leaf.dat
            if np.float64(Xleaf).ndim==0:
                if leaf.labels==j:
                    likelihood = eval_KDE(Xleaf, x, None)
                    Smyth_posts_mat[j,i]=likelihood
            elif Xleaf.ndim==1:
                class_indic = leaf.labels==T.symbols[j]
                if any(class_indic):
                    Xclass = Xleaf[class_indic]
                    likelihood = eval_KDE(Xclass, x, None)
                    Smyth_posts_mat[j,i]=likelihood*\
                        class_count/total_count
            else:
                class_indic = leaf.labels==T.symbols[j]
                if any(class_indic):
                    Xclass = Xleaf[:,class_indic]
                    likelihood = eval_KDE(Xclass, x, atts)
                    Smyth_posts_mat[j,i]=likelihood*\
                        class_count/total_count
        if any(np.isnan(Smyth_posts_mat[:,i])):
            pdb.set_trace()
        
        # normalizing Smyth's posteriors
        if sum(Smyth_posts_mat[:,i])==0.:
            Smyth_posts_mat[:,i] = emp_posts_mat[:,i]
        else:
            Smyth_posts_mat[:,i] /= sum(Smyth_posts_mat[:,i])
    return (KDE_posts_mat, emp_posts_mat, 
            Lap_posts_mat, Smyth_posts_mat)


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
    KDE_loss = (np.zeros(len(n_list)), np.zeros(len(n_list)))
    emp_loss = (np.zeros(len(n_list)), np.zeros(len(n_list)))
    Lap_loss = (np.zeros(len(n_list)), np.zeros(len(n_list)))
    Smyth_loss = (np.zeros(len(n_list)), np.zeros(len(n_list)))
    TKDE_loss = (np.zeros(len(n_list)), np.zeros(len(n_list)))
    
    accs = np.zeros((5, len(n_list)))
    for i in range(len(n_list)):
        n1, n2, n3 = n_list[i]
        X_train, Y_train, _ = gen_data.generate_class_GMM(n1, n2, n3)
        
        # training original CART tree
        print "Fitting CART tree.."
        sklearn_T = tree.DecisionTreeClassifier()
        sklearn_T.fit(np.transpose(X_train), Y_train)
        # converting it to KDE-based format
        T = tree_structure.convert_SK(sklearn_T, X_train, Y_train, 
                                      objective_builders.normal_CDF)
        # training the tree based on the KDE-based training
        print "Fitting KDE-based tree.."
        KDE_T = tree_structure.Tree(X_train, Y_train, objective_builders.normal_CDF)
        KDE_T.fit_full_tree()
        print "Pruning KDE-base tree.."
        _, best_KDE_T = fitting_tools.CV_prune(KDE_T, 5)
        
        
        print "Computing the posteriors.."
        # true posteriors
        posteriors = np.zeros((3, X_test.shape[1]))
        posteriors[0,:] = gen_data.eval_GMM(X_test, specs_list[0]) * 200. / 800.
        posteriors[1,:] = gen_data.eval_GMM(X_test, specs_list[1]) * 200. / 800.
        posteriors[2,:] = gen_data.eval_GMM(X_test, specs_list[2]) * 200. / 800.
        post_sums = np.repeat(np.expand_dims(np.sum(posteriors,axis=0),axis=0), 3, axis=0)
        posteriors /= post_sums
        # estimated posteriors
        lambdas = np.array([n1, n2, n3]) / float(n1+n2+n3)
        est_posteriors = eval_posteriors(X_test, T, lambdas) + \
            (KDE_T.posteriors_predict(X_test)[0],)
        
        # compute MSE loss of posterior estimations
        KDE_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[0])**2, axis=0)/3.) / float(n1+n2+n3)
        emp_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[1])**2, axis=0)/3.) / float(n1+n2+n3)
        Lap_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[2])**2, axis=0)/3.) / float(n1+n2+n3)
        Smyth_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[3])**2, axis=0)/3.) / float(n1+n2+n3)
        TKDE_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[4])**2, axis=0)/3.) / float(n1+n2+n3)
        
        # compute log-loss of posterior estimations
        # first, get rid of "zeros"
        posteriors[posteriors==0] = posteriors[posteriors==0] + 1e-6
        for t in range(len(est_posteriors)):
            zero_indics = est_posteriors[t]==0
            est_posteriors[t][zero_indics] = est_posteriors[t][zero_indics] + 1e-6
        KDE_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[0])))/float(n1+n2+n3)
        emp_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[1])))/float(n1+n2+n3)
        Lap_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[2])))/float(n1+n2+n3)
        Smyth_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[3])))/float(n1+n2+n3)
        TKDE_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[4])))/float(n1+n2+n3)
        
        # compute the accurac for each posterior too
        for t in range(len(est_posteriors)):
            preds = np.argmax(est_posteriors[t], axis=0)
            accs[t,i] = np.sum(T.symbols[preds] == Y_test) / float(len(Y_test))
        
    return KDE_loss, emp_loss, Lap_loss, Smyth_loss, TKDE_loss, accs
    
    
def eval_KDE(X, x_test, atts):
    """Evaluating KDE-based likelihood of a sample based on Smyth's formulation
    
    The given data matrix should be column-wise with each row indicating
    a feature. The test sample should be a 1D array.
    """
    
    X = np.float64(X)
    # if there is only a scalar data as X:
    # no need to iterative over samples
    if X.ndim==0:
        n = 1
        sigma = .5
        KDE_d = np.exp(-(X - x_test)**2/(2.*sigma**2))/\
            (np.sqrt(2.*np.pi)*sigma)
        return KDE_d
    # if there is a 1D array data as X:
    # no need to iterate over attributes
    elif X.ndim==1:
        
        n = len(X)
        # test sample should also be a scalar
        if np.float64(x_test).ndim>0:
            raise ValueError("If leaf's data is a 1D array, test"+
                             " sample can only be a scalar.")
        # Silverman's kernel width
        sigma = 1.06*np.std(X_d)*n**(-1/5.)
        KDE_d = np.exp(-(X - x_test)**2/ (2.*sigma**2))/\
            (np.sqrt(2.*np.pi)*sigma)
        
        return sum(KDE_d) / float(n)
    
    #If there is 2D array data as X
    else:
        n = X.shape[1]
        # likeliood contribtion of each sample is computed
        # by multiplying 1D kernels over those attributes 
        # that are given
        likelihoods_arr = np.ones(n)
        for d in atts:
            # only take the d'th feature
            xd_test = x_test[d]
            Xd = X[d,:]

            # Silverman kernel width in current attribute
            if np.std(Xd)==0.:
                sigma = .5
            else:
                sigma = 1.06*np.std(Xd)*n**(-1/5.)
            # computing KDE at this attribute
            KDE_d = np.exp(-(Xd - xd_test)**2/(2.*sigma**2))/\
                (np.sqrt(2.*np.pi)*sigma)
            # multiplying KDE of this attribute
            likelihoods_arr *= KDE_d

        # adding all samples' contributiong
        return sum(likelihoods_arr) / float(n)

    
    
