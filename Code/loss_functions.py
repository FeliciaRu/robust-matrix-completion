# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:46:08 2020

The class LossFunction can be extended with any kind
of loss function that is differentiable. 

See below for example loss functions.

@author: F.Ruppel
"""
import numpy as np


class LossFunction(object):
    def rho(self, x):
        """
        Rho function (loss function)
        
        Parameters
        -----------
        z : array (1d)

        Returns
        -------
        rho : array (1d)
            
        """
        raise NotImplementedError

    def psi(self, x):
        """
        Derivative of rho

        Parameters
        ----------
        z : array (1d)

        Returns
        -------
        psi : array (1d)
        """
        raise NotImplementedError
        
    def alpha(self):
        """
        Parameter alpha to obtain Fisher consistency
        during joint regression and scale estimation
        
        (see A. M. Zoubir, V. Koivunen, E. Ollila, and M. Muma, Robust
         statistics for signal processing. Cambridge University Press,
         2018., pp. 57-60)
        
        Returns
        -------
        alpha : real number

        """
        raise NotImplementedError
        
    def name(self):
        """
        Function name for labels, e.g. in plots

        Returns
        -------
        name : string

        """
        raise NotImplementedError

###################################################################
#################### Example loss functions #######################

class PseudoHuber(LossFunction):
    def rho(self, x):
        return 0.5*(abs(x)+(1+abs(x))**(-1)-1)

    def psi(self, x):
        return np.asarray((0.5*x*(abs(x)+2))/(abs(x)+1)**2)
    
    def alpha(self):
        return 2* 0.0915070106
    
    def name(self):
        return 'pseudo-Huber'   
        
class LeastSquares(LossFunction):
    
    def rho(self, x):
        return 0.5 * x * x
    
    def psi(self, x):
        return x
    
    def alpha(self):
        return 0.5
    
    def name(self):
        return 'Least squares'
    
class Huber(LossFunction):
    c = 1.345 # Constant to trade off robustness and efficiency
    
    def rho(self, x):
        x = np.asarray(x)
        subset = abs(x) <= Huber.c
        return 0.5 * x**2 * subset + (1 - subset) * (Huber.c *abs(x)  - 0.5 * Huber.c**2)

    def psi(self, x):
        x = np.asarray(x)
        subset = abs(x) <= Huber.c
        return subset * x + (1-subset) * Huber.c * np.sign(x) 
        
    def alpha(self):
        return 0.7102
    
    def name(self):
        return 'Huber'

