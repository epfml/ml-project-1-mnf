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

  
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            w=ws[n_iter]-gamma*compute_gradient(minibatch_y,minibatch_tx,ws[n_iter])
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
    """
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
    mse=compute_mse_loss(y, tx, w)

    return w, mse



######### Function implementation for logistic regression ######### 

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array """

    s=(1)/(1+np.exp(-t))
    return s



def calculate_log_likelihood_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    N=y.shape[0]
    y_pred=tx.dot(w)
    loss=-np.mean((y*np.log(sigmoid(y_pred))+(1-y)*np.log(1-sigmoid(y_pred))))

    return loss


def calculate_log_likelihood_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    N=y.shape[0]
    grad=(1/N)*((tx.T).dot((sigmoid(tx.dot(w)))-y))

    return(grad)





################################ Features expansion using polynomial regression ################################

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)
    """
    X = np.zeros((len(x),degree+1))
    for i in range (1,degree+1):
        X[:,i] = x**i
    return X






# logistic regression using classic gradient descent algorithm

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
        """
    threshold = 1e-8
    losses = []
    w=initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w=w-gamma*calculate_log_likelihood_gradient(y,tx,w)
        loss=calculate_log_likelihood_loss(y,tx,w)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("loss={l}".format(l=calculate_log_likelihood_loss(y, tx, w)))
    return(w,loss)

   


def reg_logistic_regression(y,tx,lambda_,initial_w, max_iters, gamma):
    # init parameters
    losses = []
    threshold = 1e-8

    # build tx
    w=initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        #compute the new gradient
        grad=calculate_log_likelihood_gradient(y,tx,w) + 2*lambda_*w
        w=w-gamma*grad
        loss=calculate_log_likelihood_loss(y,tx,w)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
   
    print("loss={l}".format(l=calculate_log_likelihood_loss(y, tx, w)))

    return(w,loss)



