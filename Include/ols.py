
def ols_estimate(y: np.array, x: np.ndarray):
    """
    y: np.array : n x 1 response
    x: np.ndarray : n x p design matrix
    :return: estimates ols for the model of the form y = x beta as inv(x'x)x'y
    """
    # center the vectors (they should be adjusted in the intercept later)
    y_center: np.array = y - y.mean()
    x_center: np.ndarray = x - x.mean(axis=0)

    # inv(x'x)
    d: np.ndarray = np.linalg.pinv(np.matmul(np.transpose(x_center), x_center))

    # ols beta
    beta: np.array = np.matmul(np.matmul(d, np.transpose(x)), y_center)

    # compute intercept
    intercept = y.mean() - np.dot(x.mean(axis=0), beta)

    return np.append(intercept, beta)
