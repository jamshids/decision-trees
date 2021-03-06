import numpy as np
import objective_builders
import optimizations
import pdb
import copy
import tree_structure
from objective_builders import normal_CDF
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import tree, ensemble

def split_features(leaf, kernel_CDF, selected_feat=None):
    """
    Computing the best splits of each feature by minimizing the 
    given objectives within the node rule's limits
    
    The output would be the best split of each feature and the resulting
    optimization value (scores for the split). If no feature component is
    given, do the optimization for all the features
    """
    
    if selected_feat:
        priors = leaf.class_prob
        theta, score = optimizations.minimize_KDE_entropy(X, leaf.labels, 
                                                          kernel_CDF, priors)
        return theta, score
    
    # dimensionality of data
    d = 1 if leaf.dat.ndim==1 else leaf.dat.shape[0]
    
    scores = np.zeros(d)
    thetas = np.zeros(d)
        
    # do the 1D optimization (inside the intervals permitted by the rules)
    for i in range(d):
        X = leaf.dat[i,:] if d>1 else leaf.dat
        priors = leaf.class_prob
        thetas[i], scores[i] = optimizations.minimize_KDE_entropy(X, leaf.labels, 
                                                                  kernel_CDF, priors)
        
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

def rule_divide(dat, rule):
    """
    Divide a given data into two parts based on a given rule
    """
    if dat.ndim==1:
        theta = rule['0']
        left_inds = np.where(dat <= theta)[0]
        right_inds = np.where(dat > theta)[0]
        
    elif dat.ndim==2:
        if len(rule.keys())>1:
            raise ValueError('Number of keys in a rule dictionary' + 
                             ' should always be 1')
        div_feature = rule.keys()
        theta = rule[div_feature[0]]
        left_inds = np.where(dat[int(div_feature[0]),:] <= theta)[0]
        right_inds = np.where(dat[int(div_feature[0]),:] > theta)[0]
    
    return left_inds, right_inds

def compute_probs_KDE(leaf, kernel_CDF, symbols, theta, 
                      selected_feature, uncond_sigma=None, cond_sigma=None):
    """Given that we are on a specific node, computing the probability of
    reaching left and right children. The filtered data at the selected
    variable (feature component) and the optimized threshold should be 
    among the inputs
    
    Note that this function will be used for a leaf that has been 
    divided into children. That is to say, the leaf's data cannot be
    a single scalar, otherwise it wouldn't have been divided (hence
    it is NOT dimension-less)
    """
    
    # the data restricted to the selecte feature
    d = 1 if leaf.dat.ndim==1 else leaf.dat.shape[0]
    X = leaf.dat[selected_feature,:] if d>1 else leaf.dat
    c = len(symbols)
    n = len(leaf.labels)
    
    # piors
    left_posts = np.zeros(c)
    right_posts = np.zeros(c)
        
    # marginal CDF of the kernel: Pr (x<theta | N_i)
    marginal_CDF = kernel_CDF(X, theta)
    # taking care of being 1. or 0.
    if marginal_CDF == 1.:
        marginal_CDF -= 1e-6
    elif marginal_CDF == 0.:
        marginal_CDF += 1e-6

    for j in range(c):
        # computing class-conditional marginal CDFs:
        # Pr (x<theta | y=j, N_i) 
        indic_j = leaf.labels==symbols[j]
        if np.any(indic_j):
            X_j = X[indic_j]
            cc_marginal_CDF = kernel_CDF(X_j, theta)
                        
            # computing the children's posteriors 
            left_posts[j] = leaf.class_prob[j]*(cc_marginal_CDF/
                                             marginal_CDF)
            right_posts[j] = leaf.class_prob[j]*((1-cc_marginal_CDF)/
                                              (1-marginal_CDF))
           
    # re-normalization of the posteriors to prevent accumulation 
    # of the errors
    left_posts = left_posts / np.sum(left_posts)
    right_posts = right_posts / np.sum(right_posts)
    
    # probability of reaching left node (x<=theta)
    left_reach_prob = marginal_CDF
    right_reach_prob = 1 - marginal_CDF
    
    
    return left_posts, right_posts, left_reach_prob, right_reach_prob

def CV_prune(T, n_folds):
    """Pruning a given (full tree) to the right size using cross validation
    """

    # partitioning the data into several folds
    total_labels = T.node_dict['0'].labels
    n = len(total_labels)
    total_dat = T.node_dict['0'].dat
    dim = total_dat.ndim
    cv_kfold = StratifiedKFold(n_splits = n_folds)
    train_inds, test_inds = list(), list()
    for trains, tests in cv_kfold.split(np.transpose(total_dat), total_labels):
        train_inds += [trains]
        test_inds += [tests]

    # generating the tree sequences for all the folds and the whole 
    # training data set
    all_alphas = list()
    all_seqs = list()
    seq, alphas = T.cost_complexity_seq()
    all_seqs += [copy.deepcopy(seq)]
    all_alphas += [copy.deepcopy(alphas)]
    
    # building tree sequences for each CV fold
    for i in range(n_folds):
        if dim==1:
            train_dat = total_dat[train_inds[i]]
        else:
            train_dat = total_dat[:,train_inds[i]]
        train_labels = total_labels[train_inds[i]]
        
        # create a full tree based on the training folds
        cv_T = tree_structure.Tree(train_dat, train_labels, 
                                   T.kernel_CDF)
        cv_T.fit_full_tree()
        print "CV %d .. " % i,
        
        # generate the pruned sequence of subtrees 
        seq, alphas = cv_T.cost_complexity_seq()
        all_seqs += [copy.deepcopy(seq)]
        all_alphas += [copy.deepcopy(alphas)]
        
    print "\n Now choosing the best tree.."
    # scoring for each subtrees of the main tree using 
    # cross-validation subtrees
    scores = np.zeros(len(all_seqs[0]))
    c = len(np.unique(T.symbols))
    for i in range(len(scores)-1):
        # alpha' will be the geometric mean of the current 
        # alpha value and its consequent one
        alpha_pr = np.sqrt(all_alphas[0][i]*all_alphas[0][i+1])
        
        # Among subtrees of a specific CV fold, choose the one 
        # whose alpha-value is the largest one that is less than
        # alpha' and then predict class labels of the test fold 
        # over that subtree
        misclasses_mat = np.zeros((c, n_folds))
        for cv_ind in range(n_folds):
            valid_alphas = all_alphas[1+cv_ind] <= alpha_pr
            chosen_alpha = np.argmax(np.array(all_alphas[1+cv_ind])
                                     [valid_alphas])
            chosen_sub = all_seqs[1+cv_ind][chosen_alpha]
            #  throw in the tests into the chosen subtree
            if dim==1:
                test_dat = total_dat[test_inds[cv_ind]]
            else:
                test_dat = total_dat[:, test_inds[cv_ind]]
            test_labels = total_labels[test_inds[cv_ind]]
            
            _, predictions = chosen_sub.posteriors_predict(test_dat)
            #pdb.set_trace()
            for j in range(c):
                class_inds = test_labels==T.symbols[j]
                misclasses_mat[j,cv_ind]=np.sum(predictions[class_inds] 
                                                != j) 
        
        # compute total misclassifications
        scores[i] = np.sum(misclasses_mat) / float(n)
        
    # score of the root
    scores[-1] = all_seqs[0][-1].total_misclass_rate()
    
    # choosing the best subtree (with the prefernce of subtrees
    # with higher alphas
    all_best_subs = np.where(scores==np.min(scores))[0]
    if len(all_best_subs)>1:
        best_sub = np.sort(all_best_subs)[-1]
    else:
        best_sub = all_best_subs[0]
    
    # take the tree with minimum score
    return scores, all_seqs[0][best_sub]
            
            
def create_forest(tree_num, X, Y, method):
    """Creating a forest with specified number of trees
    and based on a given training data set
    
    Method could be CART or KDE. The output format for these two
    methods will be different.
    """
    
    if method=='CART':
        forest = ensemble.RandomForestClassifier(n_estimators=tree_num)
        forest.fit(X.T, Y)
    elif method=='KDE':
        forest = []
        n = len(X) if X.ndim==1 else X.shape[1]
        for i in range(tree_num):
            # resampling
            boot_inds = np.random.randint(0, n, n)
            boot_X = X[boot_inds] if X.ndim==1 else X[:,boot_inds]
            boot_Y = Y[boot_inds]
            # tree construction
            T = tree_structure.Tree(boot_X, boot_Y, 
                                    objective_builders.normal_CDF)
            T.fit_full_tree()
            
            forest += [copy.deepcopy(T)]
            print i
    
    return forest
