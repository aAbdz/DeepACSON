# -*- coding: utf-8 -*-
"""
Created on Fri Aug 03 11:38:36 2018

@author: aliabd
"""

import numpy as np

def cart2pol(x,y):
    rho=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    phi=phi*(180/np.pi)
    return(rho,phi)

def pol2cart(rho, phi):
    phi=phi*(np.pi/180)
    x=rho*np.cos(phi)
    y=rho*np.sin(phi)
    return(x,y)