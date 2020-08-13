# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:41:39 2020

This file contains functions that perform matrix completion on a low-rank matrix
either via joint regression and scale estimation or gradient descent.

@author: F.Ruppel
"""
import numpy.random as rd
import numpy as np

import loss_functions as lf


"""
Perform matrix completion using joint regression and scale estimation.

Parameters
-----------
data:       array (2d, n1xn2), matrix to be completed. Unobserved values can be NaN.
Omega:      boolean array (2d, n1xn2), denotes observed cells
r:          desired rank
loss_fun:   loss function, derived from loss_functions.LossFunction
--Optional:
eps:        epsilon for termination condition
maxiter:    max. numer of iterations of main loop


Returns
-------
M_est:      completed matrix (2d array, n1xn2)
U:          factor U (2d array, n1xr)
V:          factor V (2d array, rxn2)
    
"""
def complete_matrix(data, Omega, r, loss_fun=None, eps=10**(-4), maxiter=10):
    # Parse input
    n1, n2 = data.shape
    
    if loss_fun == None:
       loss_fun = lf.PseudoHuber()
    
    # Initialize
    U = rd.randn(n1, r)
    V = rd.randn(r, n2)
    finished = False
    counter = 0     # counts the number of iterations
    
    
    # Optimize U and V
    while not finished:
        # update V
        for j in range(0, n2):
            # find relevant entries in data for column j
            idx = Omega[:,j]
            U_idx = U[idx,:]
            b_idx = data[idx,j] 
            
            V[:, j] = joint_regression_scale(b_idx, U_idx, loss_fun, initial_guess=V[:,j])
            
            
        # update U
        U_new = U.copy()
        for i in range(0, n1):
           # find relevant entries in data for row i
           idx = Omega[i,:]
           V_idx = V[:,idx].T
           b_idx = data[i,idx]
           
           U_new[i, :] = joint_regression_scale(b_idx, V_idx, loss_fun, initial_guess=U[i,:])
        counter += 1
        
        # Check termination conditions
        if counter >= maxiter:
            finished = True
            print('Maxiter was reached during robust matrix completion. Consider increasing it.')
        
        if np.sum(abs(U_new-U))/(n1*n2) < eps:
            finished = True
        U = U_new
            
    M_est = U.dot(V)       
    return M_est, U, V

"""
Perform matrix completion using gradient descent.

Parameters
-----------
data:       array (2d, n1xn2), matrix to be completed. Unobserved values can be NaN.
Omega:      boolean array (2d, n1xn2), denotes observed cells
r:          desired rank
loss_fun:   loss function, derived from loss_functions.LossFunction
--Optional:
eps:        epsilon for termination condition
maxiter:    max. numer of iterations of main loop
min_scale:  if estimated scale is smaller than min_scale, it is set to min_scale


Returns
-------
M_est:      completed matrix (2d array, n1xn2)
U:          factor U (2d array, n1xr)
V:          factor V (2d array, rxn2)
    
"""
def complete_matrix_GD(data, Omega, r, loss_fun=None, eps=10**(-8), maxiter=500, min_scale = 1):
    # Parse input
    n1, n2 = data.shape
    
    if loss_fun == None:
       loss_fun = lf.PseudoHuber1()
    
    # Initialize
    U = rd.randn(n1, r)
    V = rd.randn(r, n2)
    finished = False
    counter = 0
    
    # Save previous states of V, U and their gradients
    V_prev = np.zeros(V.shape)
    V_grad_prev = np.zeros(V.shape)
    U_prev = np.zeros(U.shape)
    U_grad_prev = np.zeros(U.shape)
    
    # Optimize U and V
    while not finished:
        #estimate scale
        res = data[Omega] - (U@V)[Omega]    #residual
        scale = 1.4815*np.median(abs(res-np.median(res))) 
        if scale < min_scale:
            scale = min_scale
            
        # update V
        V_grad = np.zeros(V.shape)
        for j in range(0, n2):
            idx = Omega[:,j]
            U_idx = U[idx,:]
            b_idx = data[idx,j] 

            V_grad[:, j] = getGradient(U_idx, b_idx, V[:, j], scale, loss_fun)
        if counter == 0:
            gamma = 0.005   #initial learning rate
        else:
            #estimate gamma from previous gradient
            gamma = np.linalg.norm(((V-V_prev).reshape((-1,1))).T @ ((V_grad - V_grad_prev).reshape((-1,1))))/(np.linalg.norm(V_grad - V_grad_prev)**2)
        # save states for next iteration
        V_prev = V.copy()
        V_grad_prev = V_grad.copy()
        
        V = V - gamma * V_grad
        
        #estimate scale
        res = data[Omega] - (U@V)[Omega]
        scale = 1.4815*np.median(abs(res-np.median(res))) 
        if scale < min_scale:
            scale = min_scale
            
        # update U
        U_grad = np.zeros(U.shape)
        for i in range(0, n1):
           idx = Omega[i,:]
           V_idx = V[:,idx].T
           b_idx = data[i,idx] 
           
           U_grad[i, :] = getGradient(V_idx, b_idx, U[i, :], scale, loss_fun)
        if counter == 0:
            gamma = 0.005   #initial learning rate
        else:
            #estimate gamma from previous gradient
            gamma = np.linalg.norm(((U-U_prev).reshape((-1,1))).T @ ((U_grad - U_grad_prev).reshape((-1,1))))/(np.linalg.norm(U_grad - U_grad_prev)**2)
        # save states for next iteration
        U_prev = U.copy()
        U_grad_prev = U_grad.copy()
        
        U = U - gamma * U_grad
        
        counter += 1
        
        # Check termination conditions
        if counter >= maxiter:
            finished = True
            print('Maxiter was reached during robust matrix completion (gradient descent). Consider increasing it.')

        if np.sum(abs(U-U_prev))/(n1*n2) < eps:
            finished = True
            
    M_est = U.dot(V)       
    return M_est, U, V


'''
Returns the gradient for robust regression
'''
def getGradient(Z, y, beta, sigma, loss_fun):
    return loss_fun.psi((Z@beta - y)/sigma) @ Z

    
'''
Joint regression and scale estimation.
See also M. Zoubir, V. Koivunen, E. Ollila, and M. Muma, Robust statistics
for signal processing. Cambridge University Press, 2018, isbn: 1107017416.
'''
def joint_regression_scale(b, X, loss_fun, scale= None, initial_guess=None):

    # read alpha from loss function
    alpha = loss_fun.alpha()

    gamma = 2       # for real data
    maxiter = 1500
    eps = 1e-5      # for termination condition
    counter = 0     # counts the number of iterations
    
    X_plus = np.linalg.pinv(X)
    
    if initial_guess is None:
        initial_guess = np.zeros(X.shape[1], 1)
        
    beta = initial_guess
    
    if scale is None:
        r = b - X.dot(beta)
        scale = 1.4815*np.median(abs(r-np.median(r)))   #madn estimate
    
    N = b.shape[0]
    p = beta.shape[0]
    while counter < maxiter:
        # update residuals
        r = b - X.dot(beta)
        
        # update chi of residuals
        r_chi = loss_fun.psi(r/scale) * r/scale - loss_fun.rho(r/scale)
        
        # estimate scale
        scale = np.sqrt(((gamma * scale**2) / (2*alpha*(N-p-1)))*np.sum(r_chi))
        
        # update pseudo residuals
        r_pseu = loss_fun.psi(r/scale) * scale
        
        # compute regression update
        delta = X_plus.dot(r_pseu)
        
        # compute convergence criterion
        crit = np.linalg.norm(delta)/np.linalg.norm(beta)
        
        # update beta
        beta = beta + delta
        
        # check if error < eps
        if crit < eps:
            break
    return beta