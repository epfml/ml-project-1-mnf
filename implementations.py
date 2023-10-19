"""some function that will be used in the progect"""

import numpy as np
from helpers import batch_iter


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N=y.shape[0]
    err=y-np.dot(tx,w)
    coef=-(1/N)
    gradient=coef*(np.dot(np.transpose(tx),err))
    # ***************************************************
    return gradient



def compute_mse_loss(y, tx, w):
        """Calculate the loss using either MSE or MAE.

        Args:
            y: numpy array of shape=(N, )
            tx: numpy array of shape=(N,2)
            w: numpy array of shape=(2,). The vector of model parameters.

        Returns:
            the value of the loss (a scalar), corresponding to the input parameters w.
        """
        N=tx.shape[0]
        err=(y-tx.dot(w))
        loss=(1/(2*N))*(np.transpose(err)).dot(err)
        return loss




def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        w=ws[n_iter]-gamma*compute_gradient(y,tx,ws[n_iter])
        loss=compute_mse_loss(y,tx,w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws




def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    #w = initial_w
    num_batches=max_iters
    batch_y=[]
    batch_x=[]
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches):
        batch_y.append(minibatch_y)
        batch_x.append(minibatch_tx)
   
  
    for n_iter in range(max_iters):
        
        # ***************************************************
        w=ws[n_iter]-gamma*compute_gradient(batch_y[n_iter],batch_x[n_iter],ws[n_iter])
        print(w)
        loss=compute_mse_loss(y,tx,w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1])
        )
    return losses, ws



def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    
    #print((tx.T).dot(tx))
    #print((tx.T).dot(y))
    w=np.linalg.solve((tx.T).dot(tx), (tx.T).dot(y))
    mse=compute_mse_loss(y, tx, w)
    return w, mse
    # returns mse, and optimal weights



def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    lambda_p = lambda_*2*len(y)
    w = np.linalg.solve((tx.T).dot(tx)+lambda_p*np.eye(tx.shape[1]),(tx.T).dot(y))
    return w