import tensorflow as tf

# INPUT: x is K x N x M
# Performs dropout on the input, with the same M x W dropout mask across the N
# dimension. Then rehsapes to (MN x W) and computes x*W + b
# OUTPUT: y is K x N x output_size
#         reg_loss is a scalar corresponding to a regularization loss
def dense_with_dropout(x, output_size, prob, weight_reg=0,
                       nonlinearity=(lambda x: x), is_test=False, stddev=0.05,
                       bias_start=0.0, name='DenseDropout'):
    with tf.variable_scope(name):
        (K,N,M) = x.get_shape().as_list()
        test_noise_shape = [1,N,M]

        W = tf.get_variable("W", [M, output_size], tf.float32,
                        initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size],
                        initializer=tf.constant_initializer(bias_start))

        h1 = tf.cond(is_test, lambda: tf.nn.dropout(x, keep_prob=1-prob, noise_shape=test_noise_shape),
                              lambda: tf.nn.dropout(x, keep_prob=1-prob))

        h1 = tf.reshape(h1, (-1, M))
        y = nonlinearity(tf.matmul(h1, W) + b)
        y = tf.reshape(y, (-1, N, output_size))

        reg_loss = weight_reg * tf.reduce_sum(W**2) / (1. - prob)

        return y, reg_loss


def concrete_dropout(x, p, noise_shape=None, temp=0.1):
    if not noise_shape:
        noise_shape = tf.shape(x)

    eps = 1e-9
    unif_noise = tf.random_uniform(shape=noise_shape)
    drop_prob = (
        tf.log(p + eps)
        - tf.log(1. - p + eps)
        + tf.log(unif_noise + eps)
        - tf.log(1. - unif_noise + eps)
    )
    drop_prob = tf.sigmoid(drop_prob / temp)
    drop_mask = 1. - drop_prob
    retain_prob = 1. - p
    x *= drop_mask
    x /= retain_prob
    return x


# Performs dropout on the input, with the same M x W dropout mask across the N
# dimension. Then rehsapes to (MN x W) and computes x*W + b
# OUTPUT: y is K x N x output_size
#         reg_loss is a scalar corresponding to a regularization loss term
def dense_with_concrete_dropout(x, output_size, nonlinearity=(lambda x: x),
                                weight_reg = 0,
                                dropout_reg =1e-5,
                                is_test=False, stddev=0.05, bias_start=0.0,
                                name='DenseConcreteDropout'):
    with tf.variable_scope(name):
        (K,N,M) = x.get_shape().as_list()
        test_noise_shape = [1,N,M]

        W = tf.get_variable("W", [M, output_size], tf.float32,
                        initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size],
                        initializer=tf.constant_initializer(bias_start))
        p_logit = tf.get_variable("p", [], tf.float32,
                        initializer=tf.random_uniform_initializer(-2., 0.))

        p = tf.sigmoid(p_logit)

        h1 = tf.cond(is_test, lambda: concrete_dropout(x, p, noise_shape=test_noise_shape),
                              lambda: concrete_dropout(x, p))

        h1 = tf.reshape(h1, (-1, M))
        y = nonlinearity(tf.matmul(h1, W) + b)
        y = tf.reshape(y, (-1, N, output_size))

        kernel_regularizer = weight_reg * tf.reduce_sum(W**2) / (1. - p)
        dropout_regularizer = dropout_reg * ( p * tf.log(p) + (1. - p) * tf.log(1. - p) )
        reg_loss = kernel_regularizer + dropout_regularizer

        return y, reg_loss

# Applies a sequence of dense_with_dropout layers
# INPUT: x, the input to the layer
#        hidden_layer_sizes, the widths of the hidden layers
#        dropout_prob, the probabality of dropping the input neurons
#        is_test, boolean corresponding to the phase of operation
#        weight_regularizer, the coefficient for regularizing the layer weights
# OUTPUT: current_input, the output of the MLP
#         reg_loss, the regularization loss corresponding to all the layers
def mlp_with_dropout(x, hidden_layer_sizes, dropout_prob, is_test, weight_reg=0):
    current_input = x
    reg_losses = []
    for i,s in enumerate(hidden_layer_sizes):
        dropout = 0.0 if (i==0) else dropout_prob
        current_input, reg_loss = dense_with_dropout(current_input, output_size=s,
                                       prob=dropout, weight_reg=weight_reg,
                                       nonlinearity=tf.nn.relu, is_test=is_test, name="fc%d"%(i))
        reg_losses.append(reg_loss)
        print("layer {}, shape {}".format(i, current_input.get_shape().as_list()))

    reg_loss = tf.add_n(reg_losses)
    return current_input, reg_loss

# Applies a sequence of dense_with_concrete_dropout layers
# INPUT: x, the input to the layer
#        hidden_layer_sizes, the widths of the hidden layers
#        is_test, boolean corresponding to the phase of operation
#        weight_regularizer, the coefficient for regularizing the layer weights
#        dropout_regularizer, the coefficient for regularizing the dropout probability
# OUTPUT: current_input, the output of the MLP
#         reg_loss, the regularization loss corresponding to all the layers
def mlp_with_concrete_dropout(x, hidden_layer_sizes, is_test, weight_reg=0, dropout_reg=1e-5):
    current_input = x
    reg_losses = []
    for i,s in enumerate(hidden_layer_sizes):
        # if i == 0:
        #     current_input, reg_loss = dense_with_dropout(current_input, output_size=s,
        #                                    prob=0.0, weight_regularizer=weight_regularizer,
        #                                    nonlinearity=tf.nn.relu, is_test=is_test, name="fc%d"%(i))
        # else:
        current_input, reg_loss = dense_with_concrete_dropout(current_input, output_size=s,
                                       weight_reg=weight_reg,
                                       dropout_reg=dropout_reg,
                                       nonlinearity=tf.nn.relu, is_test=is_test, name="fc%d"%(i))
        reg_losses.append(reg_loss)
        print("layer {}, shape {}".format(i, current_input.get_shape().as_list()))

    reg_loss = tf.add_n(reg_losses)
    return current_input, reg_loss

# INPUT: all inputs are K x N x M
# OUTPUT: K x N array of distances between x_hat and x in the mahalanobis sense
def mahalanobis_distance(x, x_hat, var, name="MahalanobisDist"):
    eps = 1e-7
    with tf.variable_scope(name):
        return tf.reduce_sum( (x - x_hat)**2 / (var+eps), axis=2 )

# INPUT: x has size (K x N x M)
# OUTPUT: (K x M) array of the variance along the 2nd dimension
def epistemic_unc(x, name="EpistemicUnc"):
    with tf.variable_scope(name):
        mean = tf.reduce_mean( x, axis=1, keep_dims=True )
        var = tf.reduce_mean( (x - mean)**2, axis=1 )
        return var
