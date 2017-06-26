import numpy as np
import copy
import fitting_tools

class Node:
    """
    Tree node: left and right child + data which can be any object
    """

    def __init__(self, dat, labels, is_stopped, reach_prob, class_prob,
                 rules=None, parent=None, left_child=None, right_child=None):
        """
        Node constructor
        """
        self.left = left_child
        self.right = right_child
        self.parent = parent
        self.rules = rules
        self.dat = dat
        self.labels = labels
        self.is_stopped = is_stopped
        self.reach_prob = reach_prob
        self.class_prob = class_prob
        
    def compute_error_rate(self):
        """Computing error rate of the node, based on the current posteriors
        """
        
        return 1 - max(self.class_prob)

class Tree(Node):
    """
    Tree class: it has a list of nodes, where the leaves are determined
    """
    
    def __init__(self, dat, labels, kernel_CDF):
        """
        Tree constructor: the objectives is a array of functions, each
        an objective to minimize over a single feature component
        """
        
        # the class priors (and symbols) are also the same for all the nodes, 
        #so compute it as a property of the tree (and not each node)
        # --------------------------------------
        c = len(np.unique(labels))
        is_stopped = True if c==1 else False
        class_info = np.zeros((2, c))
        class_info[0,:] = np.unique(labels)
        # comuting pies:
        for i in range(c):
            class_info[1,i] = np.sum(labels==class_info[0,i]) / len(labels)
        # --------------------------------------
        # now assigning the min-max (rules of the root) 
        if dat.ndim==1:
            a,b = (dat.min(), dat.max())
            rules = {'0': (a,b)}
        elif dat.ndim==2:
            a,b = np.min(dat, axis=1), np.max(dat, axis=1)
            rules = {}
            for i in range(len(a)):
                rules.update({str(i): (a[i], b[i])})
        else:
            raise ValueError('Dimension of the input data should not be higher than 2')
        # --------------------------------------
        # properties:
        self.node_list = [Node(dat, labels, is_stopped, 1., class_info[1,:], rules)]
        self.leaves = [self.node_list[0]]
        self.symbols = class_info[0,:]
        self.kernel_CDF = kernel_CDF
        
    
    def check_full_stopped(self):
        """
        Checking if all the leaves of the tree are stopped
        """

        for leaf in self.leaves:
            if not(leaf.is_stopped):
                return False

        # if the For-loop is not broken, it means all the leaves are
        # stopped --> return true
        return True
    
    def fit_full_tree(self):
        """
        Method for fitting a full tree (a tree with maximum purity)
        from the root node. 
        
        Starting from the root node, we keep adding nodes until each
        leaf has samlpes with only a single unique class label.
        """
        
        # start adding children until full purity is obtained
        while not(self.check_full_stopped()):
            
            # go through all the leaves and split the non-stopped ones
            leaf_list = copy.deepcopy(self.leaves)
            inds_to_remove = []
            
            for i in range(len(self.leaves)):
                
                leaf = self.leaves[i]
                
                if not(leaf.is_stopped):
                    
                    # best splits:
                    thetas, scores = fitting_tools.split_features(leaf, self.kernel_CDF)
                    selected_feature = np.argmin(scores)
                    best_split = thetas[selected_feature]
                    
                    # creating the new left/right leaves
                    # ----------------------------------------
                    # priors and reachin probabilities
                    probs = fitting_tools.compute_probs_KDE(leaf, self.kernel_CDF, self.symbols, 
                                                            best_split, selected_feature)
                    left_priors, right_priors = probs[:2]
                    left_reach_prob = probs[2] * leaf.reach_prob
                    right_reach_prob = probs[3] * leaf.reach_prob

                    # new rules:
                    left_rules, right_rules = fitting_tools.update_rules(best_split, selected_feature, leaf)
                    # new data
                    left_dat, left_labels = fitting_tools.filter_data(leaf.dat, leaf.labels, 
                                                        left_rules, selected_feature)
                    right_dat, right_labels = fitting_tools.filter_data(leaf.dat, leaf.labels, 
                                                          right_rules, selected_feature)
                    # stop the new leaves or not?
                    is_left_stopped = len(np.unique(left_labels))==1
                    is_right_stopped = len(np.unique(right_labels))==1
                    # children:
                    left_child = Node(left_dat, left_labels, is_left_stopped, 
                                      left_reach_prob, left_priors, left_rules, leaf)
                    right_child = Node(right_dat, right_labels, is_right_stopped, 
                                       right_reach_prob, right_priors, right_rules, leaf)
                    
                    # updating the tree structure
                    # ---------------------------
                    # mark the index of the current leaf to throw it out from the leaf-list
                    inds_to_remove += [i]
                    # add the new nodes as the children of the old leaf
                    node_index = self.node_list.index(leaf)
                    leaf.left_child = left_child
                    leaf.right_child = right_child
                    self.node_list[node_index] = leaf
                    # add the new leaves into the leaf-list
                    leaf_list += [left_child, right_child]
                    self.node_list += [left_child, right_child]
                
           
            leaf_list = np.delete(np.array(leaf_list), inds_to_remove).tolist()
            
            # after going through all the leaves, update the leaf list
            self.leaves = leaf_list
            
            #pdb.set_trace()

        
