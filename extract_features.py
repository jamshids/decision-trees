import numpy as np
import prep_dat
import pdb
import tensorflow as tf

def full_connected_autoencoder(Xin, size_list, learning_rate, batch_size, epochs):
    """Creating an autoencoder with multiple fully connected layers
    
    The depth and number of nodes of each layer is determined in one
    of the input arguments. The input list specifies only sizes of
    the encoder layers. Symmetircally, the decoder will use the flipped
    version of this list to build layers of the decoder part.
    """
    
    # doing autoencoding for 1-D features is meaningless
    if Xin.ndim==1:
        raise ValueError("The input array should be 2D")
    else:
        d = Xin.shape[0]
    
    print("Creating the model..")
    
    """Placeholder for the input:"""
    # The input X has dim. dxn, but the input placehoder for the
    # autoencoder is considered to be nxd to be consistent with
    # TF documentation
    X = tf.placeholder("float", [None, d])
    
    """parameters of the encoder:"""
    # array of data dimensionality in each layer
    dim_array = np.array([d] + size_list)
    depth = len(size_list)
    enc_pars = {}
    dec_pars = {}
    for i in range(depth):
        key = "w_"+str(i+1)
        enc_pars.update({key: tf.Variable(tf.random_normal([dim_array[i], 
                                                            dim_array[i+1]]))})
        dec_pars.update({key: tf.Variable(tf.random_normal([dim_array[-i-1], 
                                                            dim_array[-i-2]]))})

        key = "b_"+str(i+1)
        enc_pars.update({key: tf.Variable(tf.random_normal([dim_array[i+1]]))})
        dec_pars.update({key: tf.Variable(tf.random_normal([dim_array[-i-2]]))})
        
    """Modeling encoder and Decoder"""
    # now that the linear parameters (W,b) are formed, let's model the
    # encoder and decoder parts
    encoder = model_layers(X, enc_pars)
    decoder = model_layers(encoder, dec_pars)
    
    # reconstructed data
    reconst_X = decoder
    # original data
    original_X = X
    
    # loss function of the network
    loss = tf.reduce_mean(tf.pow(original_X - reconst_X, 2))
    # optimizer to use
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    
    """Preparing the Batches"""
    n = Xin.shape[1]
    batch_inds = prep_dat.gen_batch_inds(n, batch_size)
    batches = prep_dat.gen_batch_matrices(Xin.T, batch_inds)
    
    """Initialization and Run the Training"""
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            # go through  all the batches one-by-one
            for j in range(len(batch_inds)):
                this_batch = batches[j]
                _, c = sess.run([optimizer, loss], feed_dict={X:this_batch})
                
            print("batch %d is done" % i, 
                  "loss=", "{:.9f}".format(c))
                
    
def model_layers(Xin, pars):
    """Modeling few layers of a network, by inputting its linear parameters
    and using Sigmoid as the layers' activation function
    """
    
    depth = int(len(pars) / 2)

    # start from the first layer and evaluate output of all layers
    # iteratively
    output = tf.nn.sigmoid(tf.add(tf.matmul(Xin, pars["w_1"]), pars["b_1"]))
    for i in range(1, depth):
        output = tf.nn.sigmoid(tf.add(tf.matmul(output, 
                                                pars["w_%d" % (i+1)]), pars["b_%d" % (i+1)]))
        
    return output

    
