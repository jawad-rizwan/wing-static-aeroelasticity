# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 23:51:38 2025
AER722 - Project 1
@author: Jawad Rizwan, Peter Stefanov
"""
import math
import sympy as sp
import numpy as np

"""
CONSTANTS
--------------------------------------------------
Global variables, can be edited per question part
"""
y = sp.symbols('y')
k = 0
GJ1 = 0 # constants in Nm^2/rad
GJ2 = 0 # constants in Nm^2/rad

"""
BASELINE FORMULAS
--------------------------------------------------
"""

# CLb and CMb are assumed to be constant along span and equal to thin airfoil theory values **GET FROM THIN AIRFOIL THEORY**

#alphaBar = 5-k*eta

"""
FUNCTIONS
--------------------------------------------------
"""

def eta(y, s):
    return(y/s)

# use assumed mode method with as many modes as required in series 1=y, 2=2y^2, 3=3y^3...
def nextmode(n, y):
    return n*y**n

# find qd with error < 0.1% and plot qd vs n for each find

def error(previous, new):
    return((previous-new)/new*100)
# returns percentage error

def SS (GJ, modei, modej, y, s): #structural stiffness
    return sp.integrate(GJ*sp.diff(modei)*sp.diff(modej) ,(y, 0, s))


'''
Question a)
'''
c1 = 0.3 # m
c2 = 0.4 # m
s = 2 # m
GJ1 = 8500 # Nm^2/rad
GJ2 = 7500 # Nm^2/rad
eta = eta(y, s)

modes = []
for x in np.linspace(1, 6, 6):
    modes.append(nextmode(x, y))
print(modes)