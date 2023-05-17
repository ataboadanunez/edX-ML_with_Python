"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from scipy.stats import multivariate_normal
from common import GaussianMixture

from IPython import embed

# def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
#     """E-step: Softly assigns each datapoint to a gaussian component
#     Args:
#         X: (n, d) array holding the data
#         mixture: the current gaussian mixture
#     Returns:
#         np.ndarray: (n, K) array holding the soft counts
#             for all components for all examples
#         float: log-likelihood of the assignment
#     """
#     n, d = X.shape
#     K = mixture.mu.shape[0]
#     likelihoods = np.empty([n, K])
#     for i in range(K):
#         likelihoods[:,i] = multivariate_normal.pdf(X, mixture.mu[i,:], np.identity(d)*mixture.var[i])
#     posteriors = np.empty([n,K])
#     for i in range(K):
#         posteriors[:,i] = likelihoods[:,i]*mixture.p[i]
#     den = np.sum(posteriors, axis=1)
#     posts = (posteriors.T / den.T).T
#     log_likelihood = np.log((likelihoods*mixture.p).sum(axis=1)).sum() 
#     return posts, log_likelihood


# def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
#     """M-step: Updates the gaussian mixture by maximizing the log-likelihood
#     of the weighted dataset
#     Args:
#         X: (n, d) array holding the data
#         post: (n, K) array holding the soft counts
#             for all components for all examples
#     Returns:
#         GaussianMixture: the new gaussian mixture
#     """
#     d = X.shape[1]
#     K = post.shape[1]
#     mu = np.empty([K, d])
#     var = np.empty(K)
#     p = post.mean(axis=0)
#     for i in range(K):
#         mu[i,:] = (post[:,i] * X.T).sum(axis=1)/post[:,i].sum()
#         var[i] = ((((X - mu[i,:])**2).T*post[:,i]).sum(axis=1)/post[:,i].sum()).mean()
#     return GaussianMixture(mu, var, p)
    


# def run(X: np.ndarray, mixture: GaussianMixture,
#         post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
#     """Runs the mixture model
#     Args:
#         X: (n, d) array holding the data
#         post: (n, K) array holding the soft counts
#             for all components for all examples
#     Returns:
#         GaussianMixture: the new gaussian mixture
#         np.ndarray: (n, K) array holding the soft counts
#             for all components for all examples
#         float: log-likelihood of the current assignment
#     """
#     mix = mixture
#     new_log_likelihood = 0
#     old_log_likelihood = -np.inf
#     while abs((new_log_likelihood - old_log_likelihood)) > 1e-6*abs(new_log_likelihood):
#         old_log_likelihood = new_log_likelihood
#         post, new_log_likelihood = estep(X, mix)
#         mix = mstep(X, post)
#     return mix, post, new_log_likelihood


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    
    n, d = X.shape

    k = mixture.mu.shape[0]

    likelihoods = np.zeros((n, k))
    for i in range(k):
        likelihoods[:, i] = multivariate_normal.pdf(X, mixture.mu[i, :], np.identity(d) * mixture.var[i])

    posteriors = np.zeros((n, k))
    for i in range(k):
        posteriors[:, i] = likelihoods[:, i] * mixture.p[i]

    posteriors_sum = np.sum(posteriors, axis=1)
    posts = (posteriors.T / posteriors_sum.T).T
    log_likelihood = np.log((likelihoods * mixture.p).sum(axis=1)).sum()

    return posts, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    d = X.shape[1]
    k = post.shape[1]
    w = np.mean(post, axis=0)
    mu = np.zeros((k, d))
    sigma = np.zeros((k))

    for i in range(k):

        
        mu[i, :] = np.sum(post[:, i] * X.T, axis=1) / np.sum(post[:, i])
        diff = X - mu[i]
        sigma[i] = ((((diff)**2).T*post[:,i]).sum(axis=1)/post[:,i].sum()).mean()

    return GaussianMixture(mu, sigma, w)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    
    mix = mixture
    new_ll = 0
    old_ll = -np.inf

    while abs((new_ll - old_ll))  > 1e-6 * abs(new_ll):
        old_ll = new_ll
        post, new_ll = estep(X, mix)
        mix = mstep(X, post)
        
    return mix, post, new_ll
