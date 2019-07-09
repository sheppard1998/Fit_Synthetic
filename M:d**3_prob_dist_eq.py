#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:44:16 2019

@author: ssheppa1
"""

import numpy as np

from scipy import special

def mass_distance_prior(m, m_mean, m_dev, d, d_mean, d_dev):
    
    r = (m - m_mean)/((d - d_mean)**3)
    
    fresnelS, fresnelC = special.fresnel(m_dev/np.sqrt(np.pi*r*m_dev*d_dev**3))
    
    prob = (np.sqrt(m_dev)/((4*d_dev*r**2)**(3/2)))*(np.pi*np.cos(m_dev/ \
           (2*r*d_dev**3))*(1-2*fresnelC) + np.pi*np.sin(m_dev/(2*r*d_dev**3))*(1-2*fresnelS))
    
    return prob