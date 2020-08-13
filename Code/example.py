# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:36:00 2020

@author: F.Ruppel
"""
import numpy as np

import utils
import loss_functions as lf
import robust_matrix_completion as rmc


n1, n2, r = 150, 300, 10     # Matrix dimensions and rank
c = 0.1                      # Outlier percentage
SNR = 9
p = 0.45                     # Sampling percentage

M, U_true, V_true = utils.createMatrix(n1, n2, r)
# contaminate M
M_cont, outlier_locations = utils.contaminate_SNR(M, c, SNR, sigma_ratio = np.sqrt(1000))
# print RMSE after contamination
print('RMSE after contamination :', np.sqrt(np.sum((M-M_cont)**2)*(1/(n1*n2))))

# Obtain a sample from M_cont
Omega, data = utils.randomSample(M_cont, p)

loss_functions = [lf.PseudoHuber(), lf.Huber(), lf.LeastSquares()]

for loss_fun in loss_functions:
    # estimate M
    M_est, U, V = rmc.complete_matrix(data, Omega, r, loss_fun=loss_fun)
    # uncomment next line to use gradient descent rather than joint regression and scale estimation
    # M_est, U, V = rmc.complete_matrix_GD(data, Omega, r, loss_fun=loss_fun)
    RMSE = np.sqrt(np.sum((M-M_est)**2)*(1/(n1*n2)))
            
    print('RMSE,', loss_fun.name(),': ', RMSE)
    