# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:55:46 2020

@author: Jonathan Browning
"""

import numpy as np
from scipy.stats import gaussian_kde as kdf
from scipy import special as sp

class AlphaMu:
    numSamples = 2*(10**6)  # the number of samples used in the simulation
    r = np.linspace(0, 6, 6000) # theoretical envelope PDF x axes
    
    def __init__(self, alpha, mu, r_hat):
        
        # user input checks and assigns value
        self.alpha = self.input_Check(alpha, "\alpha", 1, 10)
        self.mu = self.input_Check(mu, "\mu", 1, 10)
        self.r_hat = self.input_Check(r_hat, "\hat{r}", 0.5, 2.5)  
        
        # simulating and theri densities 
        self.multipathFading = self.complex_Multipath_Fading()
        self.xdataEnv, self.ydataEnv = self.envelope_Density()
        
        # theoretical PDFs calculated
        self.envelopeProbability = self.envelope_PDF()

    def input_Check(self, data, inputName, lower, upper):
        # input_Check checks the user inputs
        
        # has a value been entered
        if data == "":
            raise ValueError(" ".join((inputName, "must have a numeric value")))
        
        # incase of an non-numeric input 
        try:
            data = float(data)
        except:    
            raise ValueError(" ".join((inputName, "must have a numeric value")))
    
        # data must be within the range
        if data < lower or data > upper:
            raise ValueError(" ".join((inputName, f"must be in the range [{lower:.2f}, {upper:.2f}]")))
        
        if inputName == "\mu" and data.is_integer() == False:
            raise ValueError(" ".join((inputName, "must be an integer")))
        
        return data
    
    def scattered_Component(self):
        # scattered_Component calculates the power of the scattered signal component
        
        sigma = self.r_hat**(self.alpha/2) / np.sqrt( 2 * self.mu)
        
        return sigma
    
    def generate_Gaussians(self, mean, sigma):
        # generate_Gaussians generates the Gaussian random variables
        
        gaussians = np.random.default_rng().normal(mean, sigma, self.numSamples)
        
        return gaussians
    
    def complex_Multipath_Fading(self):
        # complex_Multipath_Fading generates the complex fading random variables
        
        sigma = self.scattered_Component()
        
        multipathFading = 0
        i = 1
        while i <= (self.mu):
            X_i = self.generate_Gaussians(0, sigma)
            Y_i = self.generate_Gaussians(0, sigma)
        
            multipathFading = multipathFading + X_i**(2) + Y_i**(2)
            i+=1
            
        return multipathFading
    
    def envelope_PDF(self):
        # envelope_PDF calculates the theoretical envelope PDF
        
        A = self.alpha * self.mu**(self.mu) * self.r**((self.alpha*self.mu)-1)
        B = self.r_hat**(self.alpha*self.mu) * sp.gamma(self.mu)
        C = np.exp(-self.r**(self.alpha)*self.mu / (self.r_hat**(self.alpha)))
        PDF = (A / B) * C
        
        return PDF
    
    def envelope_Density(self):
        # envelope_Density finds the envelope PDF of the simulated random variables
        
        R = (self.multipathFading)**(1/self.alpha)
        kde = kdf(R)
        x = np.linspace(R.min(), R.max(), 100)
        p = kde(x)
        
        return x, p
