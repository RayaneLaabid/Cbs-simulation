#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from math import comb,ceil
from itertools import combinations


def gen_data(setting, n, sig): 
    """
    Generate synthetic data for simulation studies.
    
    Parameters:
    -----------
    setting : int
        Data generation scenario (1-8). Each setting defines different relationships
        between features and response, and different noise structures.
    n : int
        Number of samples to generate.
    sig : float
        Noise level/signal strength parameter.
    
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix of shape (n, 20) with features drawn from Uniform(-1, 1).
    Y : numpy.ndarray
        Response variable of length n with added noise.
    mu_x : numpy.ndarray
        True mean function (without noise) of length n.
    """
    # Generate feature matrix: n samples with 20 features each
    # Features are drawn from uniform distribution on [-1, 1]
    X = np.random.uniform(low=-1, high=1, size=n*20).reshape((n,20))
    
    if setting == 1: 
        # Setting 1: Complex interaction-based mean function
        # mu_x depends on interactions between X[:,0] and X[:,1], and thresholding on X[:,3]
        # Creates a piecewise function with different behaviors based on sign of X[:,0] * X[:,1]
        mu_x = (X[:,0] * X[:,1] > 0 ) * (X[:,3]*(X[:,3]>0.5) + 0.5*(X[:,3]<=0.5)) + (X[:,0] * X[:,1] <= 0 ) * (X[:,3]*(X[:,3]<-0.5) - 0.5*(X[:,3]>-0.5))
        mu_x = mu_x * 4  # Scale the mean function
        Y = mu_x + np.random.normal(size=n) * sig  # Add Gaussian noise
        # plt.scatter(mu_x, Y)  # Optional: visualize relationship
        return X, Y, mu_x
    
    if setting == 2:
        # Setting 2: Linear interaction + exponential term
        # mu_x = X[:,0] * X[:,1] + exp(X[:,3] - 1), scaled by 5
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig  # Higher noise multiplier
        return X, Y, mu_x
        
    if setting == 3:
        # Setting 3: Same mean function as setting 2, but heteroscedastic noise
        # Noise variance depends on the magnitude of mu_x
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 4:
        # Setting 4: Same mean function, but more complex heteroscedastic noise
        # Noise variance is a piecewise function of mu_x
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x
    
    if setting == 5:
        # Setting 5: Similar to setting 1 but without scaling
        # Uses the same complex interaction structure but different scaling
        mu_x = (X[:,0] * X[:,1] > 0 ) * (X[:,3]>0.5) * (0.25+X[:,3]) + (X[:,0] * X[:,1] <= 0 ) * (X[:,3]<-0.5) * (X[:,3]-0.25)
        mu_x = mu_x  # No additional scaling
        Y = mu_x + np.random.normal(size=n) * sig
        return X, Y, mu_x
    
    if setting == 6:
        # Setting 6: Extended mean function with quadratic term
        # Includes X[:,2]^2 term in addition to linear interaction and exponential
        mu_x = (X[:,0] * X[:,1] + X[:,2]**2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y, mu_x
    
    if setting == 7:
        # Setting 7: Same mean function as setting 6, but heteroscedastic noise
        mu_x = (X[:,0] * X[:,1] + X[:,2]**2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 8:
        # Setting 8: Same mean function as setting 6, but complex heteroscedastic noise
        mu_x = (X[:,0] * X[:,1] + X[:,2]**2 + np.exp(X[:,3] - 1) - 1) * 2
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x
     
 

def BH(pvals, nb_batch, q = 0.1):
    """
    Implements the Benjamini-Hochberg (BH) procedure for multiple testing.
    
    This function performs FDR control by comparing test scores against calibration scores
    to determine which test points should be selected (reject null hypothesis).
    
    Parameters:
    -----------
    pvals : numpy.ndarray
        Array of p-values for each batch.
    nb_batch : int
        Number of batches.
    q : float, optional (default=0.1)
        Target false discovery rate (FDR) level.
    
    Returns:
    --------
    numpy.ndarray
        Indices of test points that are selected (null hypothesis rejected).
        Empty array if no points are selected.
    """
    df_test = pd.DataFrame({"id": range(nb_batch), "pval": pvals}).sort_values(by='pval')
    df_test['threshold'] = q * np.linspace(1, nb_batch, num=nb_batch) / nb_batch 
    idx_smaller = [j for j in range(nb_batch) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    if len(idx_smaller) == 0:
        return(np.array([]))
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller)+1)])
        return(idx_sel)

def varphi_all_permutations(calib_scores, batch_size, function, level, max_samples=10000):
    """
    Optimized version of varphi_all_permutations using Monte Carlo sampling.
    
    Instead of computing all possible combinations (which is computationally infeasible),
    we use Monte Carlo sampling to approximate the distribution of varphi values.
    
    Parameters:
    -----------
    calib_scores : numpy.ndarray
        Calibration scores.
    batch_size : int
        Size of each batch.
    function : callable
        Function to apply to each batch (e.g., quantile_function).
    level : float
        Level parameter for the function.
    max_samples : int, optional (default=10000)
        Maximum number of Monte Carlo samples to use.
    
    Returns:
    --------
    numpy.ndarray
        Array of varphi values from Monte Carlo sampling.
    """
    n = len(calib_scores)
    try:
        total_combinations = comb(n+batch_size, batch_size)
        if total_combinations <= max_samples:
            num_samples = total_combinations
        else:
            num_samples = max_samples
    except OverflowError:
        num_samples = max_samples
    
    varphi_all = np.zeros(num_samples)
    for i in range(num_samples):
        batch_indices = np.random.choice(n+batch_size, batch_size, replace=False)
        batch = calib_scores[batch_indices]
        varphi_all[i] = function(batch, level)
    
    return varphi_all

def quantile_function(batch, level):
    """
    Compute the quantile of a batch at the specified level.
    
    Parameters:
    -----------
    batch : numpy.ndarray
        Array of values.
    level : float
        Quantile level (0 to 1).
    
    Returns:
    --------
    float
        The level-th quantile of the batch.
    """
    return np.sort(batch)[ceil(len(batch)*level)-1]

def pval_function(calib_varphi, hypothesis_varphi):
    """
    Compute p-value using optimized calculation.
    
    Parameters:
    -----------
    calib_varphi : numpy.ndarray
        Array of calibration varphi values.
    hypothesis_varphi : float
        Test varphi value.
    
    Returns:
    --------
    float
        P-value for the hypothesis test.
    """
    smaller_count = np.sum(calib_varphi < hypothesis_varphi)
    equal_count = np.sum(calib_varphi == hypothesis_varphi)
    
    if equal_count > 0:
        equal_count = np.random.uniform(0, equal_count)
    
    return (smaller_count + equal_count) / len(calib_varphi)

def compute_ranks(test_scores, calib_scores):
    sorted_calib = np.sort(calib_scores)
    n = len(calib_scores)
    ranks = []
    
    for score in test_scores:
        # Find the minimum rank where test score <= calibration score
        rank = np.searchsorted(sorted_calib, score, side='right')
        if rank == 0:
            rank = 1
        elif rank > n:
            rank = n + 1
        ranks.append(rank)
    
    return np.array(ranks)


def generate_rank_sequences(m, n):
    sequences = []
    for combo in combinations(range(1, n+2), m):
        sequences.append(sorted(combo))
    return sequences


def pval_pi_quantile(score_size , batch_size, rank, level ):
    M = comb(score_size+batch_size ,  batch_size)
    p_val = 0
    q_level = ceil(level*batch_size)
    for i in range(1,rank+1):
        p_val += comb(i + q_level -2 , q_level-1)* comb(score_size + batch_size-i-q_level+1, batch_size - q_level)
    
    return p_val/M
