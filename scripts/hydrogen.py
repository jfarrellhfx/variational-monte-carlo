"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020

Final project for PHY 407: Computational Physics.  Instr. Nicolas Grisouard.

This script uses the variational method to solve the 1D  hydrogen atom.
"""

# imports
import numpy as np
import random
import matplotlib.pyplot as plt

# get the problem class from vmc.py
from vmc import problem

# set number of monte carlo steps
N = 50000

# create an instance of the "problem" class from vmc.py.  This means we need
# to give it a step width for the derivatives, a step width for the markov
# chain.
hydrogen = problem(0.00001, 4, N, 1.5)

# Quick helper function to give the coulomb potential
def V(alpha, R):
    return - 1 / np.sqrt(np.sum(R**2))

# This is our variational "ansatz".  This time, the alpha variable is just a
# scalar.  As a guess, we use a gaussian function with a width (alpha) that
# we will try and dial in to get the best energy.
def psi(alpha, R):
    return np.exp(-1 * alpha * np.sqrt(np.sum(R ** 2)))

# tell the instance of the "problem" class what its ansatz is, and what the
# potential is
hydrogen.psi = psi
hydrogen.V = V

### Main Loop -----------------------------------------------------------------
# define a storage list for the energies
energies = []
standard_dev = []

# create a list of different values of "alpha" we will try.  We choose 10 values
# between 0.5 and 2.0.  The exact answer should be alpha = 1.
alphas = np.linspace(0.1, 1.5, 25)

# go through each value of alpha
for alpha in alphas:

    # initialize the location
    R = np.array([1, 1, 1])

    # tell the problem object which alpha we are working on
    hydrogen.alpha = alpha

    estimates = np.array([])
    print("Working on alpha = {}".format(alpha))
    for i in range(N):
        R = hydrogen.monte_carlo_step(R)
        EL = hydrogen.EL(R)
        estimates = np.append(estimates, EL)
    energies.append(np.mean(estimates))
    standard_dev.append(np.std(estimates))

energies = np.array(energies)
standard_dev = np.array(standard_dev)

np.savez("hydrogen_atom.npz", alphas = alphas, energies = energies, standard_dev = standard_dev)

plt.plot(alphas, energies)
plt.show()