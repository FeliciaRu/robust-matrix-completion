# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:35:21 2020

Some utility functions.

@author: F.Ruppel
"""
import numpy as np
import numpy.random as rd

def createMatrix(n1, n2, r):
    U_true = rd.randn(n1, r)
    V_true = rd.randn(r, n2)
    M = U_true.dot(V_true)
    return M, U_true, V_true

# Contaminate with a Gaussian mixture model with the specified SNR
def contaminate_SNR(M, c, SNR=9, sigma_ratio = np.sqrt(1000)):
    n1, n2 = M.shape
    
    P_signal = np.var(M) # signal power
    P_noise = P_signal*10**(-SNR/10) # noise power
    sigma_total = np.sqrt(P_noise)
    
    sigma_N = sigma_total/np.sqrt((1-c)+c*sigma_ratio**2)
    sigma_S = sigma_ratio*sigma_N
    
    # add dense noise
    M_cont = np.copy(M)
    N = sigma_N*rd.randn(n1, n2)
    M_cont += N
    
    #add outliers
    outlier_locations = []
    for i in range(0, n1):
        for j in range(0, n2):
            rand_val = rd.rand(1)
            if rand_val <= c:
                M_cont[i,j] += sigma_S*rd.randn(1)[0]
                outlier_locations.append((i,j))
    return M_cont, outlier_locations


# Each cell has probability 'percentage' to be drawn
def randomSample(M, percentage):
    data = np.copy(M)
    n1, n2 = data.shape
    for i in range(0, n1):
        for j in range(0, n2):
            rand_val = rd.rand(1)
            if rand_val > percentage:
                data[i,j] = np.nan
    Omega = ~np.isnan(data)
    return Omega, data
