#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:07:05 2022

@author: ying

This script performs a simulation study comparing different conformal prediction methods
for false discovery rate (FDR) control in regression settings. The simulation evaluates
three regression methods (Gradient Boosting, Random Forest, SVM) across different
scenarios and compares various calibration approaches for the Benjamini-Hochberg (BH) procedure.

OPTIMIZATION NOTES:
- Uses Monte Carlo sampling instead of exhaustive permutation computation
- Vectorized operations for better performance
- Configurable sampling parameters for speed vs accuracy trade-off
"""

import numpy as np
import pandas as pd 
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from utils import gen_data, BH, quantile_function, varphi_all_permutations, pval_function, fast_varphi_estimation , compute_ranks , generate_rank_sequences,pval_pi_quantile

# Parse command line arguments for simulation parameters
# sig_id: index for signal strength parameter (0-9, corresponding to sigma values 0.1 to 1.0)
# nt_id: index for number of test samples (0-3, corresponding to [10, 100, 500, 1000])
# set_id: data generation setting/scenario (1-8, different data distributions)
# q: FDR control level (divided by 10 from command line, so 1 = 0.1, 5 = 0.5, etc.)
# nb_batch: number of batches for processing
# level: quantile level for conformal prediction
# seed: random seed for reproducibility
sig_id = int(sys.argv[1]) - 1
nt_id = int(sys.argv[2]) - 1
set_id = int(sys.argv[3])  
q = int(sys.argv[4]) / 10 
nb_batch = int(sys.argv[5])
level = int(sys.argv[6])/10
seed = int(sys.argv[7])

# Simulation parameters
n = 1000  # Number of training and calibration samples
ntests = [10, 100, 500, 1000]  # Different test set sizes to evaluate
ntest = ntests[nt_id]  # Select test set size based on nt_id
sig_seq = np.linspace(0.1, 1, num = 10)  # Signal strength values (noise level)
sig = sig_seq[sig_id]  # Select signal strength based on sig_id
reg_names = ['gbr', 'rf', 'svm']  # Names of regression methods

# Batch processing parameters
min_batch = 5
max_batch = 25
batch_sizes = []
index_batch = [0]
remaining = ntest

# Create batches with random sizes between min_batch and max_batch
while remaining > 0:
    size = min(random.randint(min_batch, max_batch), remaining)
    batch_sizes.append(size)
    index_batch.append(size+index_batch[-1]-1)
    remaining -= size

# Optimization parameters
MAX_SAMPLES = 5000  # Maximum Monte Carlo samples for varphi computation
USE_FAST_ESTIMATION = True  # Use bootstrap sampling for even faster computation

# Initialize results dataframe to store all simulation outcomes
all_res = pd.DataFrame()

# Create output directory for results
out_dir = "../results/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print("Output diretory created!")
    
# Set random seed for reproducibility
random.seed(seed)

print(f"n={n}, ntest={ntest}, sig={sig}, out_dir={out_dir}, q={q}, level={level}, seed={seed}, nb_batch={nb_batch}, set_id={set_id}, sig_id={sig_id}, nt_id={nt_id}")
print(f"Optimization: MAX_SAMPLES={MAX_SAMPLES}, USE_FAST_ESTIMATION={USE_FAST_ESTIMATION}")

# Main simulation loop: evaluate each regression method
for reg_method in range(3):
    reg_method = reg_method + 1  # Convert to 1-based indexing
    reg_name = reg_names[reg_method - 1] 
    
    print(f"\n--- Running regression method: {reg_name} (method id: {reg_method}) ---")
    
    # Generate data for training, calibration, and testing
    # Each call to gen_data returns (X, Y, mu) where:
    # - X: feature matrix
    # - Y: response variable (with noise)
    # - mu: true mean function (without noise)
    Xtrain, Ytrain, mu_train = gen_data(set_id, n, sig)
    print(f"Generated Xtrain shape: {Xtrain.shape}, Ytrain shape: {Ytrain.shape}, mu_train shape: {mu_train.shape}")
    Xcalib, Ycalib, mu_calib = gen_data(set_id, n, sig)
    print(f"Generated Xcalib shape: {Xcalib.shape}, Ycalib shape: {Ycalib.shape}, mu_calib shape: {mu_calib.shape}")
    Xtest, Ytest, mu_test = gen_data(set_id, ntest, sig)
    print(f"Generated Xtest shape: {Xtest.shape}, Ytest shape: {Ytest.shape}, mu_test shape: {mu_test.shape}")
    
    # Train the prediction model based on the selected regression method
    # Note: We're training to predict P(Y > 0) rather than the continuous Y values
    if reg_method == 1:
        print("Training GradientBoostingRegressor...")
        regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
    if reg_method == 2:
        print("Training RandomForestRegressor...")
        regressor = RandomForestRegressor(max_depth=5, random_state=0)
    if reg_method == 3:
        print("Training SVR...")
        regressor = SVR(kernel="rbf", gamma="scale")  # Fixed: use "scale" instead of 0.1
    
    # Train model to predict binary outcome: 1 if Y > 0, 0 otherwise
    regressor.fit(Xtrain, 1*(Ytrain>0))
    print("Model training complete.")
    
    # Calculate different types of calibration scores for conformal prediction
    # These scores represent the "nonconformity" or prediction error
    
    # Standard residual-based scores: Y - predicted probability
    calib_scores = Ycalib - regressor.predict(Xcalib) 
    print(f"calib_scores (first 5): {calib_scores[:5]}")
    
    # Scores for negative examples only (Y <= 0): -predicted probability
    # This focuses calibration on the null hypothesis cases
    calib_scores0 = - regressor.predict(Xcalib) 
    print(f"calib_scores0 (first 5): {calib_scores0[:5]}")
    
    # Clipped scores: only consider positive Y values, otherwise use prediction
    calib_scores_clip = Ycalib * (Ycalib > 0) - regressor.predict(Xcalib)
    print(f"calib_scores_clip (first 5): {calib_scores_clip[:5]}")
    
    # Binary clipped scores: use large constant (1000) for positive Y, prediction for negative
    # This creates a clear separation between positive and negative cases
    calib_scores_2clip = 1000 * (Ycalib > 0) - regressor.predict(Xcalib)
    print(f"calib_scores_2clip (first 5): {calib_scores_2clip[:5]}")
     
    # Test scores: negative predicted probabilities
    # We want to reject null hypothesis when test scores are small
    test_scores = Ytest - regressor.predict(Xtest) 
    test_scores0 = Ytest - regressor.predict(Xtest) 
    test_scores_clip = Ytest * (Ytest > 0) - regressor.predict(Xtest)
    test_scores_2clip = 1000 * (Ytest > 0) - regressor.predict(Xtest)

    # Initialize arrays for batch processing results
    hypothesis_test = np.zeros(nb_batch)
    pvals = np.zeros(nb_batch) 
    pvals0 = np.zeros(nb_batch)
    pvals_clip = np.zeros(nb_batch)
    pvals_2clip = np.zeros(nb_batch)
    pvals_pi = np.zeros(nb_batch)
    pvals_pi0 = np.zeros(nb_batch)
    pvals_pi_clip = np.zeros(nb_batch)
    pvals_pi_2clip = np.zeros(nb_batch)
    test_quantiles = np.zeros((nb_batch, 4))  # Store 4 different quantile types
    ranks_test_quantiles = np.zeros((nb_batch, 4))
    print(f"Processing {nb_batch} batches with optimization...")


    rank_hypothesis = compute_ranks([0],calib_scores)[0]
    rank_hypothesis0  = compute_ranks([0],calib_scores)[0]
    rank_hypothesis_clip = compute_ranks([0],calib_scores_clip)[0] 
    rank_hypothesis_2clip = compute_ranks( [0],calib_scores_2clip)[0] 
    
    for i in range(nb_batch):
        print(f"Processing batch {i+1}/{nb_batch}")
        batch_size = batch_sizes[i]
        '''
        calib_varphi = varphi_all_permutations(calib_scores, batch_size, quantile_function, level, max_samples=MAX_SAMPLES)
        calib_varphi0 = varphi_all_permutations(calib_scores0, batch_size, quantile_function, level, max_samples=MAX_SAMPLES)
        calib_varphi_clip = varphi_all_permutations(calib_scores_clip, batch_size, quantile_function, level, max_samples=MAX_SAMPLES)
        calib_varphi_2clip = varphi_all_permutations(calib_scores_2clip, batch_size, quantile_function, level, max_samples=MAX_SAMPLES)
        '''
        # Compute test quantiles for current batch
        batch_start = index_batch[i]
        batch_end = index_batch[i+1]
        
        quantil_test = quantile_function(test_scores[batch_start:batch_end], level)
        quantil_test0 = quantile_function(test_scores0[batch_start:batch_end], level)
        quantil_test_clip = quantile_function(test_scores_clip[batch_start:batch_end], level)
        quantil_test_2clip = quantile_function(test_scores_2clip[batch_start:batch_end], level)
        
        # Store all quantiles for this batch
        test_quantiles[i] = [quantil_test, quantil_test0, quantil_test_clip, quantil_test_2clip]
        ranks_test_quantiles[i] = [compute_ranks([test_quantiles[i][0]], calib_scores)[0] ,compute_ranks([test_quantiles[i][1]], calib_scores0)[0], compute_ranks([test_quantiles[i][2]], calib_scores_clip)[0] , compute_ranks([test_quantiles[i][3]], calib_scores_2clip)[0]]
        print(f"test_quantiles: {test_quantiles[i]}")
        print(f"ranks_test_quantiles: {ranks_test_quantiles[i]}")

        '''
        pvals[i] = pval_function(calib_varphi, 0)
        pvals0[i] = pval_function(calib_varphi0, 0)
        pvals_clip[i] = pval_function(calib_varphi_clip, 0)
        pvals_2clip[i] = pval_function(calib_varphi_2clip, 0)
        print(f"pvals: {pvals[i]}, pvals0: {pvals0[i]}, pvals_clip: {pvals_clip[i]}, pvals_2clip: {pvals_2clip[i]}")
        '''

        pvals_pi[i] = pval_pi_quantile(n,batch_size,rank_hypothesis,level)
        pvals_pi0[i] = pval_pi_quantile(n,batch_size, rank_hypothesis0,level)
        pvals_pi_clip[i] = pval_pi_quantile(n,batch_size, rank_hypothesis_clip,level)
        pvals_pi_2clip[i] = pval_pi_quantile(n,batch_size, rank_hypothesis_2clip,level)

        print(f"pvals_pi:{pvals_pi[i]}")


    
    print("Calling BH with pvals...")
    # BH_res= BH(pvals, nb_batch, q )
    BH_res_pi= BH(pvals_pi, nb_batch, q )

    #print(f"BH_res indices: {BH_res}")
    print(f"BH_res_pi indices: {BH_res_pi}")

    '''
    BH_res_fdp = 0  
    BH_res_power = 0
    if len(BH_res) == 0:
        print("No selections in BH_res.")
    else:
        for i in BH_res:
            BH_res_fdp += (test_quantiles[i][0] <= 0) 
            BH_res_power +=(test_quantiles[i][0] > 0) 
        BH_res_fdp  = BH_res_fdp/len(BH_res)
        BH_res_power = BH_res_power / (sum(test_quantiles[:][0] > 0) or 1)
        print(f"BH_res_fdp: {BH_res_fdp}, BH_res_power: {BH_res_power}")
    '''
    BH_res_fdp_pi = 0  
    BH_res_power_pi = 0
    if len(BH_res_pi) == 0:
        print("No selections in BH_res.")
    else:
        for i in BH_res_pi:
            BH_res_fdp_pi += (test_quantiles[i][0] <= 0) 
            BH_res_power_pi +=(test_quantiles[i][0] > 0) 
        BH_res_fdp_pi  = BH_res_fdp_pi/len(BH_res_pi)
        BH_res_power_pi = BH_res_power_pi / (sum(test_quantiles[:][0] > 0) or 1)
        print(f"BH_res_fdp: {BH_res_fdp_pi}, BH_res_power: {BH_res_power_pi}")
    
    print("Calling BH with pvals0...")
    
    #BH_rel = BH(pvals0, nb_batch, q )
    BH_rel_pi = BH(pvals_pi0, nb_batch, q )

    #print(f"BH_rel indices: {BH_rel}")
    print(f"BH_rel_pi indices: {BH_rel_pi}")

    '''
    BH_rel_fdp = 0
    BH_rel_power = 0
    if len(BH_rel) == 0:
        BH_rel_fdp = 0
        BH_rel_power = 0
        print("No selections in BH_rel.")
    else:
        for i in BH_rel:
            BH_rel_fdp += (test_quantiles[i][1] <= 0) 
            BH_rel_power += (test_quantiles[i][1] > 0) / (sum(test_quantiles[:][1] > 0) or 1)
        BH_rel_fdp = BH_rel_fdp / len(BH_rel)
        BH_rel_power = BH_rel_power / (sum(test_quantiles[:][1] > 0) or 1)
        print(f"BH_rel_fdp: {BH_rel_fdp}, BH_rel_power: {BH_rel_power}")
    '''

    BH_rel_fdp_pi = 0  
    BH_rel_power_pi = 0
    if len(BH_rel_pi) == 0:
        print("No selections in BH_res.")
    else:
        for i in BH_rel_pi:
            BH_rel_fdp_pi += (test_quantiles[i][0] <= 0) 
            BH_rel_power_pi +=(test_quantiles[i][0] > 0) 
        BH_rel_fdp_pi  = BH_rel_fdp_pi/len(BH_rel_pi)
        BH_rel_power_pi = BH_rel_power_pi / (sum(test_quantiles[:][0] > 0) or 1)
        print(f"BH_rel_fdp: {BH_rel_fdp_pi}, BH_res_power: {BH_rel_power_pi}")



    print("Calling BH with pvals_clip...")
    
    #BH_clip = BH(pvals_clip, nb_batch, q )
    BH_clip_pi = BH(pvals_pi_clip, nb_batch, q)
    #print(f"BH_clip indices: {BH_clip}")
    print(f"BH_clip_pi incidces : {BH_clip_pi}")
    '''
    BH_clip_fdp = 0
    BH_clip_power = 0
    if len(BH_clip) == 0:
        print("No selections in BH_clip.")
    else:
        for i in BH_clip:
            BH_clip_fdp += (test_quantiles[i][2] <= 0) 
            BH_clip_power += (test_quantiles[i][2] > 0) / (sum(test_quantiles[:][2] > 0) or 1)
        BH_clip_fdp = BH_clip_fdp / len(BH_clip)
        BH_clip_power = BH_clip_power / (sum(test_quantiles[:][2] > 0) or 1)
        print(f"BH_clip_fdp: {BH_clip_fdp}, BH_clip_power: {BH_clip_power}")
    '''

    BH_clip_fdp_pi = 0  
    BH_clip_power_pi = 0
    if len(BH_clip_pi) == 0:
        print("No selections in BH_res.")
    else:
        for i in BH_clip_pi:
            BH_clip_fdp_pi += (test_quantiles[i][0] <= 0) 
            BH_clip_power_pi +=(test_quantiles[i][0] > 0) 
        BH_clip_fdp_pi  = BH_clip_fdp_pi/len(BH_clip_pi)
        BH_clip_power_pi = BH_clip_power_pi / (sum(test_quantiles[:][0] > 0) or 1)
        print(f"BH_res_fdp: {BH_clip_fdp_pi}, BH_res_power: {BH_clip_power_pi}")



    print("Calling BH with pvals_2clip...")
    #BH_2clip = BH(pvals_2clip, nb_batch, q )
    
    BH_2clip_pi=BH(pvals_pi_2clip , nb_batch , q)
    #print(f"BH_2clip indices: {BH_2clip}")
    print(f"BH_pi_2clip: {BH_2clip_pi}")
    '''
    BH_2clip_fdp = 0
    BH_2clip_power = 0
    if len(BH_2clip) == 0:
        print("No selections in BH_2clip.")
    else:
        for i in BH_2clip:
            BH_2clip_fdp += (test_quantiles[i][3] <= 0) 
            BH_2clip_power += (test_quantiles[i][3] > 0) / (sum(test_quantiles[:][3] > 0) or 1)
        BH_2clip_fdp = BH_2clip_fdp / len(BH_2clip)
        BH_2clip_power = BH_2clip_power / (sum(test_quantiles[:][3] > 0) or 1)
        print(f"BH_2clip_fdp: {BH_2clip_fdp}, BH_2clip_power: {BH_2clip_power}")
    '''
    BH_2clip_fdp_pi = 0  
    BH_2clip_power_pi = 0
    if len(BH_2clip_pi) == 0:
        print("No selections in BH_res.")
    else:
        for i in BH_2clip_pi:
            BH_2clip_fdp_pi += (test_quantiles[i][0] <= 0) 
            BH_2clip_power_pi +=(test_quantiles[i][0] > 0) 
        BH_2clip_fdp_pi  = BH_2clip_fdp_pi/len(BH_2clip_pi)
        BH_2clip_power_pi = BH_2clip_power_pi / (sum(test_quantiles[:][0] > 0) or 1)
        print(f"BH_res_fdp: {BH_2clip_fdp_pi}, BH_res_power: {BH_2clip_power_pi}")

    '''
    all_res = pd.concat((all_res, 
                         pd.DataFrame({'BH_res_fdp': [BH_res_fdp], 
                                       'BH_res_power': [BH_res_power],
                                       'BH_res_nsel': [len(BH_res)],
                                       'BH_rel_fdp': [BH_rel_fdp], 
                                       'BH_rel_power': [BH_rel_power], 
                                       'BH_rel_nsel': [len(BH_rel)],
                                       'BH_clip_fdp': [BH_clip_fdp], 
                                       'BH_clip_power': [BH_clip_power], 
                                       'BH_clip_nsel': [len(BH_clip)],
                                       'BH_2clip_fdp': [BH_2clip_fdp], 
                                       'BH_2clip_power': [BH_2clip_power], 
                                       'BH_2clip_nsel': [len(BH_2clip)],
                                       'q': [q], 'regressor': [reg_name],
                                       'seed': [seed], 'sigma': [sig], 'ntest': [ntest]})))'''

#print(f"\nAll results shape: {all_res.shape}")
#print(f"Saving results to: ../results/prob_set{str(set_id)}q{str(int(q*10))}sig{str(sig_id)}nt{str(nt_id)}seed{str(seed)}.csv")

#all_res.to_csv("../results/prob_set"+str(set_id)+"q"+str(int(q*10))+"sig"+str(sig_id)+"nt"+str(nt_id)+"seed"+str(seed)+".csv")
