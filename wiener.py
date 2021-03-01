import numpy as np
from numpy.random import normal
from scipy.optimize import minimize
from matplotlib.pyplot import hist, plot


def f(x):
    return np.exp(x)


def integrate_dWI(func, upper_limit=1, lower_limit=0, sample_size=10000, granularity=1000):
    step_2 = (upper_limit - lower_limit)/granularity
    step = np.sqrt(step_2)
    sample = normal(0, 1, (granularity, sample_size)) * step
    sim = np.zeros(sample_size)

    for j in range(sample_size):
        ss = f(0)
        inc = 0
        for i in range(granularity):
            sim[j] += ss * sample[i][j]
            inc += step_2
            ss = func(inc)
    return sim


def integrate_dWW(func, sample_size=10000, granularity=1000):
    step = np.sqrt((upper_limit - lower_limit)/granularity)
    sample = normal(0, 1, (granularity, sample_size)) * step
    sim = np.zeros(sample_size)

    for j in range(sample_size):
        ss = f(0)
        inc = 0
        for i in range(granularity):
            sim[j] += ss * sample[i][j]
            inc += sample[i][j]
            ss = func(inc)
    return sim


def vasicek_interest(alpha, beta, R0, sigma,time,sample = 1000):
    exp = np.exp(-1 * beta * time)
    deterministic = exp * R0 + (alpha/beta) * (1 - exp)
    multiplier = sigma * exp
    return deterministic + multiplier * integrate_dWI(lambda x: np.exp(beta * x),upper_limit=time,sample_size=sample)


# s = vasicek_interest(0.02, 0.005, 0.001, 0.1, 2)


def CIR_iterate(alpha, beta, Ri, sigma, t, granularity = 1000):
    step = t/granularity
    exp = np.exp(-1*beta*step)
    return Ri * exp + (alpha/beta) * (1 - exp) + sigma * exp * np.sqrt(Ri) * normal(0, 1) * np.sqrt(step)


def CIR(alpha, beta, R0, sigma,t, granularity = 1000):
    step = t/granularity
    for i in range(granularity):
        if i == 0:
            x = CIR_iterate(alpha, beta, R0, sigma,t, granularity)
        else:
            xi = CIR_iterate(alpha, beta, x, sigma, t, granularity)
            x = xi
    return x


def CIR_sample(alpha, beta, R0, sigma,t=1, sample = 1000):

    sim = np.zeros(sample)

    for j in range(sample):
        sim[j] += CIR(alpha, beta, R0, sigma,t)
    return sim


s = CIR_sample(0.02, 0.005, 0.01, 0.1, t = 0.1)
hist(s,bins=100)
s = CIR_sample(0.02, 0.005, 0.01, 0.1, t = 0.9)
hist(s,bins=100)
s = CIR_sample(0.02, 0.005, 0.01, 0.1, t = 2)
hist(s,bins=100)


import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
#stock price paths (geometric)
# dS = a S dt + b S dW
# a, b constants
a = 2.4
b = 1.7
holder = np.zeros(shape=(100, 100))
S0 = 1
time = holder.shape[1]
step = 1/time
siz = holder.shape[0]
for i in range(time):
    if i == 0:
        for j in range(siz):
            holder[j, i] = S0
    else:
        dW = normal(0, 1, size=siz) * np.sqrt(step)
        for j in range(siz):
            holder[j, i] = holder[j, i-1] * (1 + a * step + b * dW[j])

fig, ax = plt.subplots(figsize=(10, 10))
for i in range(siz):
    ax.plot(holder[i, :])

plt.xlim([0,1])
plt.ylim([0.5,1.5])


# call payoff sensitivity
from scipy.stats import norm

def bsm_call(tau, price, strike, rate, vol):
    tau_sqrt = np.sqrt(tau)
    d_plus = (1/(vol * tau_sqrt)) * (np.log(price/strike) + tau * (rate - 0.5 * vol**2))
    d_minus = d_plus - vol * tau_sqrt
    return price * norm.cdf(d_plus) - np.exp(-1 * rate * tau) * strike * norm.cdf(d_minus)



tau, price, strike, rate, vol = 0.5, 1, 1.2, 0.01, 0.03

vol = np.linspace(0.9, 0.001, 100)
holder = np.zeros((100,))
for i in range(holder.shape[0]):
    holder[i] = bsm_call(tau, price, strike, rate, vol[i])

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(vol, holder)
#plt.xlim([0,2])








