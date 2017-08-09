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
    
    # confusion factor to be used in Ling's method
    s = .3
    
    # first, compute the KDE-based  posteriors at once
    KDE_posts_mat,_ = T.posteriors_predict(X)
    
    # Second, compute other posteriors, one-by-one
    emp_posts_mat =  np.zeros((c,n))
    Lap_posts_mat =  np.zeros((c,n))
    Smyth_posts_mat = np.zeros((c,n))
    Ling_posts_mat = Ling_posterior_estimation(T, X, s)
        
    for i in range(n):
        x = X[i] if X.ndim==1 else X[:,i]
        # extracting the leaf which x belongs to
        leaf_ind, path = T.extract_leaf(x)
        # attributes included in the path
        atts = []
        for t in range(len(path)-1):
            atts += [int(T.node_dict[str(path[t])].rule.keys()[0])]
        atts = np.unique(atts)
        
        leaf = T.node_dict[str(leaf_ind)]
        for j in range(c):
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
                if leaf.labels==T.symbols[j]:
                    likelihood = eval_KDE(Xleaf, x, None)
                    prior = T.node_dict['0'].class_prob[j]
                    Smyth_posts_mat[j,i]=likelihood * prior
            elif Xleaf.ndim==1:
                class_indic = leaf.labels==T.symbols[j]
                if any(class_indic):
                    Xclass = Xleaf[class_indic]
                    likelihood = eval_KDE(Xclass, x, None)
                    prior = T.node_dict['0'].class_prob[j]
                    Smyth_posts_mat[j,i]=likelihood * prior
            else:
                class_indic = leaf.labels==T.symbols[j]
                if any(class_indic):
                    Xclass = Xleaf[:,class_indic]
                    likelihood = eval_KDE(Xclass, x, atts)
                    prior = T.node_dict['0'].class_prob[j]
                    Smyth_posts_mat[j,i]=likelihood * prior
                    
        if any(np.isnan(Smyth_posts_mat[:,i])):
            pdb.set_trace()
        
        # normalizing Smyth's posteriors
        if sum(Smyth_posts_mat[:,i])==0.:
            Smyth_posts_mat[:,i] = emp_posts_mat[:,i]
        else:
            Smyth_posts_mat[:,i] /= sum(Smyth_posts_mat[:,i])
            
    return (KDE_posts_mat, emp_posts_mat, 
            Lap_posts_mat, Smyth_posts_mat, Ling_posts_mat)


def Ling_posterior_estimation(T, X, s):
    """A posterior estimation method proposed by Ling & Yan (ICML, 2003)
    
    This method computes the posterior for each test sample by 
    considering all the leaves. Contribution of each leaf depends on
    how many of the rules in the path from the root to that leaf are
    satisfied by the test sample. So in this method, we need to go through
    all the nodes and check if the test sample satisfies their rules.
    
    Here, the input variable 0<s<1 is the confusion factor introduced in
    the paper. Smaller s results less contribution from othe leaves.
    And s=1 leads to equal contribution for all the leaves.
    """
    
    # number of test samples
    n = len(X) if X.ndim==1 else X.shape[1]
    
    # posteriors in all the leaves
    c = len(T.symbols)
    all_leaf_posts = np.zeros((c,len(T.leaf_inds)))
    for t, leaf_ind in enumerate(T.leaf_inds):
        leaf = T.node_dict[str(leaf_ind)]
        all_leaf_posts[:,t] = leaf.class_prob
        
    # listing (encoding) rules of all the nodes
    # .. and also checking if each test sample satisfies them
    rules_list = []
    all_nodes = [int(node_ind) for node_ind in T.node_dict.keys()]
    all_nodes.sort()
    test_passing = np.empty((n, len(all_nodes)))
    for node_ind in all_nodes:
        if node_ind==0: continue
        node = T.node_dict[str(node_ind)]
        # determine if it's left or right child
        parent = T.node_dict[str(node.parent)]
        if parent.left==node_ind:
            sgn = 1.
        else:
            sgn = -1.
        feature = int(parent.rule.keys()[0])
        theta = parent.rule[str(feature)]
        
        # checking rule passing of the test samples
        if X.ndim==1:
            X_feat = X
        else:
            X_feat = X[feature,:]
        test_passing[:,node_ind-1] = ~(sgn*X_feat <= sgn*theta)
        
    # now that we have contribution to all the leaves, 
    # we can compute the posterior
        
    # Now, based on the passing test for all the nodes, 
    # compute for each leaf's path, how many rules are 
    # satisfied by the test samples. 
    test_leaf_diffs = np.zeros((n, len(T.leaf_inds)))
    for i, leaf_ind in enumerate(T.leaf_inds):
        path = T.node_path(leaf_ind)
        for node_ind in path[1:]:
            # when a numerical array is added to a logical array
            # each element will be added by one, if the corresponding
            # element is True
            test_leaf_diffs[:, i] += test_passing[:, node_ind-1]
    test_leaf_diffs = s**test_leaf_diffs
    norm_term = np.diag(1. / np.sum(test_leaf_diffs, axis=1))
    posteriors = np.dot(test_leaf_diffs, all_leaf_posts.T)
    posteriors = np.dot(norm_term, posteriors)
    
    return posteriors.T
    

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
    forest_loss = (np.zeros(len(n_list)), np.zeros(len(n_list)))
    Smyth_loss = (np.zeros(len(n_list)), np.zeros(len(n_list)))
    Ling_loss = (np.zeros(len(n_list)), np.zeros(len(n_list)))
    TKDE_loss = (np.zeros(len(n_list)), np.zeros(len(n_list)))
    
    accs = np.zeros((6, len(n_list)))
    AUCs = np.zeros((6, len(n_list)))
    for i in range(len(n_list)):
        n1, n2, n3 = n_list[i]
        X_train, Y_train, _ = gen_data.generate_class_GMM(n1, n2, n3)
        
        # training original CART tree and random forest
        print "Fitting CART tree.."
        sklearn_T = tree.DecisionTreeClassifier()
        sklearn_T.fit(np.transpose(X_train), Y_train)
        print "Creating a random forest"
        tree_num = 100
        forest = fitting_tools.create_forest(tree_num, X_train, Y_train)
        # converting it to KDE-based format
        T = tree_structure.convert_SK(sklearn_T, X_train, Y_train, 
                                      objective_builders.normal_CDF)
        # training the tree based on the KDE-based training
        print "Fitting KDE-based tree.."
        KDE_T = tree_structure.Tree(X_train, Y_train, objective_builders.normal_CDF)
        KDE_T.fit_full_tree()
        
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
        
        # MSE loss
        KDE_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[0])**2, axis=0)/3.) / float(n1+n2+n3)
        emp_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[1])**2, axis=0)/3.) / float(n1+n2+n3)
        Lap_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[2])**2, axis=0)/3.) / float(n1+n2+n3)
        Smyth_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[3])**2, axis=0)/3.) / float(n1+n2+n3)
        Ling_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[4])**2, axis=0)/3.) / float(n1+n2+n3)
        TKDE_loss[0][i] = np.sum(np.sum((posteriors - est_posteriors[5])**2, axis=0)/3.) / float(n1+n2+n3)
        
        # log-loss
        # first, get rid of "zeros"
        posteriors[posteriors==0] = posteriors[posteriors==0] + 1e-6
        for t in range(len(est_posteriors)):
            zero_indics = est_posteriors[t]==0
            est_posteriors[t][zero_indics] = est_posteriors[t][zero_indics] + 1e-6
        KDE_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[0])))/float(n1+n2+n3)
        emp_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[1])))/float(n1+n2+n3)
        Lap_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[2])))/float(n1+n2+n3)
        Smyth_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[3])))/float(n1+n2+n3)
        Ling_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[4])))/float(n1+n2+n3)
        TKDE_loss[1][i] = np.sum(posteriors*(np.log(posteriors) - np.log(est_posteriors[5])))/float(n1+n2+n3)
        
        # AUCs
        for t in len(est_posteriors):
            scores = est_posteriors[t]
            AUCs[t,i] = ranking_AUC(scores, Y_test)
        
        # compute the accurac for each posterior too
        for t in range(len(est_posteriors)):
            preds = np.argmax(est_posteriors[t], axis=0)
            accs[t,i] = np.sum(T.symbols[preds] == Y_test) / float(len(Y_test))
        
        
    return KDE_loss, emp_loss, Lap_loss, Smyth_loss, Ling_loss, TKDE_loss, accs, AUCs
    
    
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
        sigma = 1.06*np.std(X)*n**(-1/5.)
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

def forest_posterior(forest, X, priors):
    """Computing posterior (class) probability of given samples
    based on a random forest trained using Scikit-Learn
    
    The posteriors at each tree are computed using empirical and 
    Laplace methods. The final posterior probability of the
    forest is the average posteriors of all the trees.
    
    Note that in the Laplace correction, we use priors based on
    empirical estimation of the original label set, not the 
    re-sampled data set.
    """
    
    # getting all the trees in the forest
    Ts = forest.estimators_
    tree_num = len(Ts)
    
    # for each given sample, find its leaf in each of the trees
    leaves = forest.apply(X.T)
    
    c = forest.n_classes_
    n = leaves.shape[0]
    emp_posteriors = np.zeros((c, n))
    Lap_posteriors = np.zeros((c, n))
    
    for i in range(n):
        Ts_emp_posterior = np.zeros((c, tree_num))
        Ts_Lap_posterior = np.zeros((c, tree_num))
        
        for j in range(tree_num):
            # class fraction at the extraced leaf of this tree
            class_fracs = Ts[j].tree_.__getstate__()['values'][leaves[i,j]]
            # empirical posterior of this tree
            Ts_emp_posterior[:, j] = class_fracs / np.sum(class_fracs)
            # empirical posterior of this tree
            Ts_Lap_posterior[:,j] = (class_fracs + priors) / \
                (np.sum(class_fracs) + 1.)
            
        # taking the average of the tree's posterior
        emp_posteriors[:,i] = np.mean(Ts_emp_posterior, axis=1)
        Lap_posteriors[:,i] = np.mean(Ts_Lap_posterior, axis=1)
        
    return emp_posteriors, Lap_posteriors

def ranking_AUC(scores, labels):
    """Estimating AUC value for a set of scores and their labels
    
    The function considers all pairs of labels if multiple classes
    exist. For each pair, the scores are ranked and AUC is 
    estimated using a counting-based metric for measuring
    separability of the scores of one class versus the other.
    """
    
    # number of classes
    symbols = np.unique(labels)
    c = len(symbols)
    
    AUC = 0.
    for j in range(c):
        for k in range(j+1, c):
            # rank the scores
            sort_inds = np.argsort(scores) + 1
            sorted_Y = labels[sort_inds-1]
            
            Sj = np.sum(sort_inds[labels==symbols[j]])
            Sk = np.sum(sort_inds[labels==symbols[k]])
            nj = np.sum(labels==symbols[j])
            nk = np.sum(labels==symbols[k])
            
            AUC_jk = (Sj - nj*(nj+1)/2.) / float(nj*nk)
            AUC_kj = (Sk - nk*(nk+1)/2.) / float(nj*nk)
            
            # add the AUC estimation of this pair to the total
            AUC += max(AUC_jk, AUC_kj)
            
    # taking the average
    AUC /= c*(c-1)/2.
    
    return AUC

