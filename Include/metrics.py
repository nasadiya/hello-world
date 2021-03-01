
def mape(y: np.array, x: np.array, beta: np.array, tol: np.float64 = 1e-10):
    """
    :param y: observations
    :param x: predictors
    :param beta: estimates
    :param tol: zero threshold
    :return: prediction error
    """
    nonzero_val = (np.abs(y).min() < tol).__invert__()
    y = y[nonzero_val]
    x = x[nonzero_val, :]

    return np.abs(np.divide(y - (x @ beta[1:]) - beta[0], y)).sum()


def dist(y: np.array, x: np.array, beta: np.array):
    """
    :param y: observations
    :param x: predictors
    :param beta: estimates
    :return: prediction error
    """

    return np.linalg.norm(y - (x @ beta[1:]) - beta[0] - y)
