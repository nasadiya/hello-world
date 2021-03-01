import numpy as np
from scipy.optimize import minimize

np.random.seed(seed=30112020)


def lasso(y: np.array, x: np.ndarray):
    """
    y: np.array : n x 1 response
    x: np.ndarray : n x p design matrix
    :return: wrapper function returns lasso estimates for the best CV error
    """
    l, s = cv_error_lasso(y, x)
    return lasso_grad(y, x, l[s.min() == s][0]), s.min(), l, s


def lasso_grad(y: np.array, x: np.ndarray, lamb: np.float64 = np.float64(1)):
    """
    y: np.array : n x 1 response
    x: np.ndarray : n x p design matrix
    lamb: np.float64 : regularisation
    model is of the form y = x beta
    :return: Using Gradient descent from
    https://davidrosenberg.github.io/mlcourse/Archive/2018/Homework/hw2.pdf
    """
    # center the vectors (they should be adjusted in the intercept later)
    y_center: np.array = y - y.mean()
    x_center: np.ndarray = x - x.mean(axis=0)

    # implement the algorithm ###############################

    # set initial values for the weights
    # shrinkage adjusted dispersion inverse
    g: np.ndarray = np.linalg.pinv(np.matmul(np.transpose(x_center), x_center) + lamb * np.identity(x.shape[1]))
    # compute the initial beta weights
    weights: np.array = np.matmul(g, np.matmul(np.transpose(x_center), y_center))
    enter_loop: bool = False
    weights_new = weights
    # initial x squared column sums
    x_center_2_col: np.ndarray = 2 * np.square(x_center).sum(axis=0)

    while not convergence(weights_new, weights) or not enter_loop:
        enter_loop = True
        weights: np.array = weights_new
        # update beta
        # find optimal basis that minimises gradient
        grad: np.array = 2 * np.matmul(np.transpose(x_center), y_center - np.matmul(x_center, weights)) + \
            x_center_2_col * weights
        weights_new: np.array = soft_thresh(grad, x_center_2_col, lamb)

    # compute intercept
    intercept = y.mean() - np.dot(x.mean(axis=0), weights_new)

    return np.append(intercept, weights_new)


def soft_thresh(c: np.array, a: np.array, lamb: np.float64):
    """
    :param c: NA
    :param a: NA
    :param lamb: NA
    :return: result of soft thresholding between c/a and lamb/a
    """
    u: np.array = np.divide(c, a)
    v: np.array = lamb / a
    return np.sign(u) * np.fmax(0, u.__abs__() - v)


def cv_error_lasso(y: np.array, x: np.ndarray):
    """
    :param y: np.array : n x 1 response
    :param x: np.ndarray : n x p design matrix
    :return:
    """
    data_size: np.int64 = x.shape[0]

    # fix range for lambda
    lambda_min: np.int64 = 1
    lambda_max: np.int64 = 500
    lambda_count = (lambda_max - lambda_min) + 1
    lambda_factor = 20

    # divide data into k groups
    groups: int = 5
    # get the CV groups
    cv_groups: np.array = _srswor(data_size, groups)

    # compute loss for each observation
    # allocate space for each lambda and CV error
    s_error: np.array = np.zeros(shape=(lambda_count,))
    lamb_seq: np.array = np.zeros(shape=(lambda_count,))

    for lamb in range(lambda_count):
        lamb_seq[lamb] = lambda_factor * (lambda_min + lamb) / lambda_max
        error = 0
        for i_group in range(groups):
            # the subset on which loss is to be computed
            subset: np.array = (i_group == cv_groups)
            # the subset on which model is to be build
            model_subset: np.array = np.invert(subset)
            lasso_estimate: np.array = lasso_grad(y[model_subset], x[model_subset, :], lamb_seq[lamb])
            # compute loss
            error += np.square(y[subset] - lasso_estimate[0] - np.matmul(x[subset, :], lasso_estimate[1:])).sum()
        # mse for the lambda
        s_error[lamb] = error / data_size

    return lamb_seq, s_error


def _srswor(sample_size: np.int64, groups: np.int64 = 5):
    """
    :sample_size siz: sample size
    :groups: number of groups
    :return: np array with group allocations
    """
    # assign a U(0,1) to all elements and scale them to group size
    ran_uni: np.array = (np.random.uniform(low=0, high=0.999999, size=sample_size) * groups).astype(np.int64)
    return ran_uni


def convergence(l_1: np.array, l_0: np.array, tol: np.float64 = 1e-5):
    """
    :param l_1: parameter value at step t
    :param l_0: parameter value at step t - 1
    :param tol: tolerance
    :return: bool output for the test metric ||l_1 - l_0|| / || l_0||
    """
    con: bool = False

    norm_diff = np.linalg.norm((l_1 - l_0))
    norm_base = np.linalg.norm(l_0)

    if norm_base == 0.0:
        norm_base = np.float64(1)

    if norm_diff/norm_base < tol:
        con: bool = True

    return con

