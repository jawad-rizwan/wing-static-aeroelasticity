# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 23:51:38 2025
AER722 - Project 1
@author: Jawad Rizwan, Peter Stefanov
"""
import math
import sympy as sp
import numpy as np
import scipy.linalg as la

"""
CONSTANTS
--------------------------------------------------
Global variables, can be edited per question part
"""
y = sp.Symbol('y')
s = sp.Symbol('s')
k = 0
c1 = 0.3 # m
c2 = 0.4 # m
s_val = 2 # m - numerical value of span
GJ1 = 8500 # Nm^2/rad
GJ2 = 7500 # Nm^2/rad

"""
BASELINE FORMULAS
--------------------------------------------------
"""

# CLb and CMb are assumed to be constant along span and equal to thin airfoil theory values **GET FROM THIN AIRFOIL THEORY**

cr = c1 + c2 # Root chord length
ct = 1.75*c1 # Tip chord length

#alphaBar = 5-k*eta

"""
FUNCTIONS
--------------------------------------------------
"""

def eta(y, s):
    return(y/s)

# chord length variation along span
def c(y, s, cr, ct): 
    return cr - (cr - ct)*eta(y,s)

# distance from elastic axis to aerodynamic center
def ec(y, s, c1, cr, ct): 
    chord = c(y, s, cr, ct)
    return chord - c1 * 0.25*chord

# lift curve slope
def CLa(y, s): 
    return 2*sp.pi*sp.sqrt(1 - (eta(y,s))**2)
    
# use assumed mode method with as many modes as required in series 1=y, 2=2y^2, 3=3y^3...
def nextmode(n, y):
    return n*y**n

# Torsional stiffness variation along span
def GJ(y, s):
    eta_val = eta(y, s)
    return sp.Piecewise(
        (GJ1*(1 - 0.25*eta_val), eta_val <= 0.5),
        (GJ2*(1 - 0.2*eta_val), eta_val > 0.5)
    )

def SS (modei, modej, y, s): #structural stiffness
    GJ_val = GJ(y, s)
    return sp.integrate(GJ_val*sp.diff(modei)*sp.diff(modej),(y, 0, s))

def AS (modei, modej, y, s): #aerodynamic stiffness
    # Calculate all parameters inside the function
    chord = c(y, s, cr, ct)
    e = ec(y, s, c1, cr, ct)
    cla = CLa(y, s)
    return -1*sp.integrate(chord*e*cla*modei*modej,(y, 0, s))

def error(previous, new):
    return((previous-new)/new*100) # returns percentage error

'''
Question a)
'''
# Finding the first 5 modes (use integer mode numbers to avoid float exponents)
modes = []
for x in range(1, 6):  
    modes.append(nextmode(x, y))
print(modes)

print("\nMode functions:")
for i, prime in enumerate(modes, 1):
    print(f"Mode {i} function: {prime}")

# Finding the derivatives of the first 6 modes
modePrimes = []
for mode in modes:
    modePrimes.append(sp.diff(mode, y))

print("\nMode derivatives:")
for i, prime in enumerate(modePrimes, 1):
    print(f"Mode {i} derivative: {prime}")

# Test mode 1
print("\nTest mode 1")
test_ss = SS(modes[0], modes[0], y, s)
test_as = AS(modes[0], modes[0], y, s)

# Convert symbolic results to numeric by substituting s and evaluating
ss_val = float(test_ss.subs(s, s_val).evalf())
as_val = float(test_as.subs(s, s_val).evalf())

# build 1x1 numeric matrices
E_num = np.array([[ss_val]], dtype=float)
K_num = np.array([[as_val]], dtype=float)

# MATLAB: eig(test_ss, -test_as) -> generalized eigenproblem A v = lambda B v with B = -test_as
eigvals = la.eigvals(E_num, -K_num)
print("Eigenvalues:", eigvals)
