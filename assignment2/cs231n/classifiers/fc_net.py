from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        verbose = False
        if verbose: print('num_layers:', self.num_layers, '\n')
        len_hidden_dims = len(hidden_dims)
        for layer_num in range(1, self.num_layers+1):
          number_of_nodes = None
          if verbose: print('layer_num:', layer_num)
          if layer_num == 1: # First Layer
            if verbose: print('\tfirst_layer')
            self.params[f"W{layer_num}"] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dims[0]))
            self.params[f"b{layer_num}"] = np.zeros(hidden_dims[0], )
            if self.normalization == "batchnorm":
              self.params[f"gamma{layer_num}"] = np.ones((hidden_dims[0], ))
              self.params[f"beta{layer_num}"] = np.zeros((hidden_dims[0], ))
          elif layer_num == self.num_layers: #Last Layer
            if verbose: print('\tlast_layer')
            self.params[f"W{layer_num}"] = np.random.normal(0.0, weight_scale, (hidden_dims[-1], num_classes))
            self.params[f"b{layer_num}"] = np.zeros(num_classes, )
          else: # Hidden Layers
            if verbose: print('\thidden_layer')
            hidden_dim_curr = hidden_dims[layer_num-2]
            hidden_dim_next = hidden_dims[layer_num-1]
            self.params[f"W{layer_num}"] = np.random.normal(0.0, weight_scale, (hidden_dim_curr, hidden_dim_next))
            self.params[f"b{layer_num}"] = np.zeros(hidden_dim_next, )
            if self.normalization == "batchnorm":
              self.params[f"gamma{layer_num}"] = np.ones((hidden_dim_next, ))
              self.params[f"beta{layer_num}"] = np.zeros((hidden_dim_next, ))
            
          if verbose: 
            print(f"\tW{layer_num}:", self.params[f"W{layer_num}"].shape)
            print(f"\tb{layer_num}:", self.params[f"b{layer_num}"].shape)
            if f"gamma{layer_num}" in self.params:
              print(f"\tgamma{layer_num}:", self.params[f"gamma{layer_num}"].shape)
              print(f"\tbeta{layer_num}:", self.params[f"beta{layer_num}"].shape)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        affine_cache = None
        bn_cache = None
        relu_cache = None
        dropout_cache = None

        caches = {}
        input_data = X
        for layer_num in range(1, self.num_layers):
          weights = self.params[f"W{layer_num}"]
          biases = self.params[f"b{layer_num}"]
          temp_out, affine_cache = affine_forward(input_data, weights, biases)
          #batch/layer norm
          if self.normalization == "batchnorm":
            x = temp_out
            gamma = self.params[f"gamma{layer_num}"]
            beta = self.params[f"beta{layer_num}"]
            bn_param = self.bn_params[layer_num-1]
            temp_out, bn_cache = batchnorm_forward(x, gamma, beta, bn_param)
          relu_out, relu_cache = relu_forward(temp_out)
          #dropout
          input_data = relu_out
          cache = (affine_cache, bn_cache, relu_cache, dropout_cache) 
          caches[f"cache{layer_num}"] = cache
        
        layer_num = self.num_layers
        weights = self.params[f"W{layer_num}"]
        biases = self.params[f"b{layer_num}"]
        affine_out, affine_cache = affine_forward(input_data, weights, biases)
        caches[f"cache{layer_num}"] = affine_cache
        scores = affine_out

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dout = softmax_loss(scores, y)
        
        layer_num = self.num_layers

        w = self.params[f"W{layer_num}"]
        cache = caches[f"cache{layer_num}"]
        dx, dw, db = affine_backward(dout, cache)
        grads[f"W{layer_num}"] = dw + (self.reg * w)
        grads[f"b{layer_num}"] = db
        loss += 0.5 * self.reg * (np.sum(w * w))

        for layer_num in range(self.num_layers-1, 0, -1):
          cache = caches[f"cache{layer_num}"]
          w = self.params[f"W{layer_num}"]
          affine_cache, bn_cache, relu_cache, dropout_cache = cache
          temp_dout = relu_backward(dx, relu_cache)
          
          if self.normalization == "batchnorm":
            temp_dout, dgamma, dbeta = batchnorm_backward_alt(temp_dout, bn_cache)
          
          dx, dw, db = affine_backward(temp_dout, affine_cache)

          grads[f"W{layer_num}"] = dw + (self.reg * self.params[f"W{layer_num}"])
          grads[f"b{layer_num}"] = db
          
          if self.normalization == "batchnorm":
            grads[f"gamma{layer_num}"] = dgamma
            grads[f"beta{layer_num}"] = dbeta
          
          loss += 0.5 * self.reg * (np.sum(w * w))
        
        
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
