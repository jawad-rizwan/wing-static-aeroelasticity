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
eta = sp.Symbol('eta')
k = 0
c1 = 0.3 # m
c2 = 0.4 # m
s = 2 # m - numerical value of span
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

# chord length variation along span
def c(eta, cr, ct): 
    return cr - (cr - ct)*eta

# distance from elastic axis to aerodynamic center
def ec(eta, c1, cr, ct): 
    chord = c(eta, cr, ct)
    return chord - c1 - 0.25*chord

# lift curve slope
def CLa(eta): 
    return 2*math.pi*sp.sqrt(1 - eta**2)
    
# use assumed mode method with as many modes as required in series 1=y, 2=2y^2, 3=3y^3...
def nextmode(n, eta):
    return n*eta**n

def SS (modei, modej, GJ1, GJ2, eta, s): #structural stiffness
    integrand1 = sp.integrate(GJ1*(1 - 0.25*eta)*sp.diff(modei)*sp.diff(modej),(eta, 0, 0.5))
    integrand2 = sp.integrate(GJ2*(1 - 0.2*eta)*sp.diff(modei)*sp.diff(modej),(eta, 0.5, 1))
    return (1/s)*(integrand1 + integrand2)

def AS (modei, modej, eta, s): #aerodynamic stiffness
    # Calculate all parameters inside the function
    chord = c(eta, cr, ct)
    ecfun = ec(eta, c1, cr, ct)
    cla = CLa(eta)
    func = chord*ecfun*cla*modei*modej
    return float(s*(sp.integrate(func,(eta, 0, 1))).evalf())

def error(previous, new):
    return((previous-new)/new*100) # returns percentage error

'''
Question a)
'''
# Finding the first 5 modes (use integer mode numbers to avoid float exponents)
modes = []
for x in range(1, 6):  
    modes.append(nextmode(x, eta))
print(modes)

print("\nMode functions:")
for i, prime in enumerate(modes, 1):
    print(f"Mode {i} function: {prime}")

# Finding the derivatives of the first 6 modes
modePrimes = []
for mode in modes:
    modePrimes.append(sp.diff(mode, eta))

print("\nMode derivatives:")
for i, prime in enumerate(modePrimes, 1):
    print(f"Mode {i} derivative: {prime}")

# Test mode 1
print("\nTest mode 1")
test_ss = float(SS(modes[0], modes[0], GJ1, GJ2, eta, s).evalf())
test_as = float(AS(modes[0], modes[0], eta, s))  # AS already returns float after our previous modification

print("SS value:", test_ss)
print("AS value:", test_as)
mat_ss = np.array([[test_ss]])
mat_as = np.array([[test_as]])

# Calculate just the eigenvalues using numerical values
eigenvalues = la.eigvals(mat_ss, mat_as)  # Note the negative sign before test_as
print("\nEigenvalues:", eigenvalues)
