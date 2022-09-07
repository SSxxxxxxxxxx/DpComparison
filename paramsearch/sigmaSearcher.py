import numpy as np
from scipy.stats import norm
import datetime
from functools import lru_cache

def get_gdp_sigma(epsilon,n,epochs,batch_size, target_delta):
    steps_per_epoch = n // batch_size
    T = epochs * steps_per_epoch
    p = batch_size / n
    upper,lower=1000,0.3
    def delta(sigma):
        mu = p * np.sqrt(T * (np.exp(1 / sigma ** 2) - 1))
        return norm.cdf(-epsilon / mu + mu / 2, loc=0, scale=1) - np.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2, loc=0, scale=1)
    assert delta(upper)<=target_delta, f'{delta(upper)}, {target_delta}'
    assert delta(lower)>target_delta
    for _ in range(100):
        mid = (upper+lower)/2
        if delta(mid)>target_delta:
            lower = mid
        else:
            upper = mid
    return (upper+lower)/2

def get_gdp_eps(sigma,n,epochs,batch_size, target_delta):
    steps_per_epoch = n // batch_size
    steps = epochs * steps_per_epoch
    p = batch_size / n
    mu = p * np.sqrt(steps * (np.exp(1 / sigma ** 2) - 1))

    for eps in np.arange(0.001, 100, 0.001):
        dlt = norm.cdf(-eps / mu + mu / 2, loc=0, scale=1) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2, loc=0, scale=1)
        if dlt <= target_delta:
            return eps
    for eps in np.arange(100, 100000, 1):
        dlt = norm.cdf(-eps / mu + mu / 2, loc=0, scale=1) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2, loc=0, scale=1)
        if dlt <= target_delta:
            return eps
    raise Exception("Error: Could not find suitable privacy parameters!")


def get_rdp_sigma(epsilon,n,epochs,batch_size, target_delta):
    from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent
    steps_per_epoch = n // batch_size
    T = epochs * steps_per_epoch
    p = batch_size / n
    upper,lower=1000,0.01
    def eps(sigma):
        orders = [1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 1200))
        rdp = compute_rdp(p, sigma, T, orders)
        e,_,order = get_privacy_spent(orders, rdp, target_delta=target_delta)
        return e

    assert eps(upper)<=epsilon
    assert eps(lower)>epsilon
    best = 100000000000
    for _ in range(15):
        mid = (upper+lower)/2
        e = eps(mid)
        if round(e-best,4)==0:
            return mid
        if abs(e-epsilon)<abs(best-epsilon):
            best = e
        if e>epsilon:
            lower = mid
        else:
            upper = mid
    return (upper+lower)/2

def get_rdp_eps(sigma,n,epochs,batch_size, target_delta):
    from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent
    steps_per_epoch = n // batch_size
    steps = epochs * steps_per_epoch
    p = batch_size / n
    orders = [1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 1200))
    rdp = compute_rdp(p, sigma, steps, orders)
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=target_delta)
    return eps


def get_dp_eps(sigma,n,epochs,batch_size,target_delta):
    return epochs * np.sqrt(2 * np.log(1.25 * epochs / target_delta)) / sigma
def get_dp_sigma(epsilon,n,epochs,batch_size,target_delta):
    return epochs * np.sqrt(2 * np.log(1.25 * epochs / target_delta)) / epsilon



def _optimise(sigma, n,epochs,batch_size,delta, meth = None):
    assert meth is not None
    upper,lower=1000,0.001
    for _ in range(40):
        assert meth(upper, n,epochs,batch_size,delta) <= sigma
        assert meth(lower, n,epochs,batch_size,delta) >= sigma
        mid = (upper + lower) / 2
        if meth(mid, n,epochs,batch_size,delta) > sigma:
            lower = mid
        else:
            upper = mid
    return (upper + lower) / 2


def get_advcmp_sigma(epsilon, n,epochs,batch_size,delta):
    return np.sqrt(epochs * np.log(2.5 * epochs / delta)) * (np.sqrt(np.log(2 / delta) + 2 * epsilon) + np.sqrt(np.log(2 / delta))) / epsilon

def get_advcmp_eps(sigma, n,epochs,batch_size,delta):
    return _optimise(sigma, n,epochs,batch_size,delta,meth=get_advcmp_sigma)


def get_zcdp_sigma(epsilon, n,epochs,batch_size,delta):
    return  np.sqrt(epochs / 2) * (np.sqrt(np.log(1 / delta) + epsilon) + np.sqrt(np.log(1 / delta))) / epsilon

def get_zcdp_eps(sigma, n,epochs,batch_size,delta):
    return _optimise(sigma, n,epochs,batch_size,delta,meth=get_zcdp_sigma)
