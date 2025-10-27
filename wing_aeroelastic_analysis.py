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
import matplotlib.pyplot as plt

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

# Max number of modes
nmax = 5

# Array to store dynamic pressures and errors for each mode count
qd = np.zeros(nmax)
modeError = np.zeros(nmax)

# Values at convergence
qdconv = 0
errconv = 0
Kconv = []
Econv = []
Fconv = []

# Finding the first 5 modes (use integer mode numbers to avoid float exponents)
modes = []
for x in range(1, 6):  
    modes.append(nextmode(x, eta))
print(modes)

print("\nMode functions:")
for i, prime in enumerate(modes, 1):
    print(f"Mode {i} function: {prime}")

# Finding the derivatives of the first 5 modes
modePrimes = []
for mode in modes:
    modePrimes.append(sp.diff(mode, eta))

print("\nMode derivatives:")
for i, prime in enumerate(modePrimes, 1):
    print(f"Mode {i} derivative: {prime}")

# Test mode 1
print("\nTest for Mode 1:")
test_ss = float(SS(modes[0], modes[0], GJ1, GJ2, eta, s).evalf())
test_as = float(AS(modes[0], modes[0], eta, s))  # AS already returns float after our previous modification

print("SS value for Mode 1:", test_ss)
print("AS value for Mode 1:", test_as)
mat_ss = np.array([[test_ss]])
mat_as = np.array([[test_as]])

# Calculate just the eigenvalues using numerical values
eigenvalues = (la.eigvals(mat_ss, mat_as)).real  # Note the negative sign before test_as
print("Dynamic Pressure Mode 1:", eigenvalues)

# Calculate the first 5 modes check for convergence
for n in range(5):
    # Create empty stiffness matrices
    E = np.zeros((n+1, n+1))
    K = np.zeros((n+1, n+1))
    
    for i in range(n+1):      
        for j in range(n+1):  
            E[i, j] = float(SS(modes[i], modes[j], GJ1, GJ2, eta, s).evalf())
            K[i, j] = float(AS(modes[i], modes[j], eta, s))
            
    # Calculate divergence dynamic pressure
    eigenvalues = la.eigvals(E, K)
    real_parts = eigenvalues.real
    qd[n] = np.min(real_parts)
    
    # Check if error is less than 0.1% to stop early
    if n > 0:
        modeError[n] = error(qd[n-1], qd[n])
        if abs(modeError[n]) < 0.1:
            print(f"\nConverged at mode count {n+1} with dynamic pressure: {qd[n]}")  
            
            # Save converged values
            qdconv = qd[n]
            errconv = modeError[n]
            Kconv = K
            Econv = E
            
            break

# Print final dynamic pressures for each mode count
print("Final Dynamic Pressures for each mode count:")
for i in range(n+1):
    print(f"Mode {i+1}: {qd[i]} | Error: {modeError[i]}%")
    
# Plot the convergence of dynamic pressure vs number of modes
plt.figure(figsize=(10, 6))

# Create array of mode counts
mode_counts = np.arange(1, n+2)  # n+2 because n is 0-indexed and we want 1,2,3,4,5

# Plot dynamic pressure vs modes
plt.plot(mode_counts, qd[:n+1], 'bo-', linewidth=2, markersize=8, label='Divergence Dynamic Pressure')
if qdconv != 0:
    plt.axhline(y=qdconv, color='r', linestyle='--', linewidth=1.5, label=f'Converged value: {qdconv:.4f} Pa')

plt.xlabel('Number of Modes', fontsize=12)
plt.ylabel('Divergence Dynamic Pressure (Pa)', fontsize=12)
plt.title('Divergence Dynamic Pressure vs Number of Modes', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(mode_counts)

plt.tight_layout()
plt.show()