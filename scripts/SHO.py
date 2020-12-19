"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020

Final project for PHY 407: Computational Physics.  Instr. Nicolas Grisouard.

This script uses the variational method to solve the 1D quantum harmonic
oscillator.  To do so, we use the tools from the vmc.problem class that I
wrote in vmc.py.

"""

# imports
import numpy as np
import random
import matplotlib.pyplot as plt

# get the problem class from vmc.py
from vmc import problem

# set number of monte carlo steps
N = 100000

# create an instance of the "problem" class from vmc.py.  This means we need
# to give it a step width for the derivatives, a step width for the markov
# chain.
sho = problem(0.001, 2, N, 1.5)

# Quick helper function to give the harmonic oscillator potential (1D, with
# m = \omega = 1)
def V(alpha, R):
    return + 1 / 2 * R ** 2

# This is our variational "ansatz".  This time, the alpha variable is just a
# scalar.  As a guess, we use a gaussian function with a width (alpha) that
# we will try and dial in to get the best energy.
def psi(alpha, R):
    return np.exp(-alpha ** 2 * R **2 / 2)

# tell the instance of the "problem" class what its ansatz is, and what the
# potential is
sho.psi = psi
sho.V = V

### Main Loop -----------------------------------------------------------------
# define a storage list for the energies
energies = []
standard_dev = []

# create a list of different values of "alpha" we will try.  We choose 10 values
# between 0.5 and 2.0.  The exact answer should be alpha = 1.
alphas = np.linspace(0.5, 1.5, 15)

# go through each value of alpha
for alpha in alphas:

    # initialize the location at the origin (has to be numpy array, because
    # of the way I wrote it; this is inconvenient now, but will generate nicely
    # to higher dimensions
    R = np.array([0])

    # tell the problem object which alpha we are working on
    sho.alpha = alpha


    estimates = np.array([])
    print("Working on alpha = {}".format(alpha))
    for i in range(N):
        R = sho.monte_carlo_step(R)
        EL = sho.EL(R)
        estimates = np.append(estimates, EL)
    energies.append(np.mean(estimates))
    standard_dev.append(np.std(estimates))
energies = np.array(energies)
standard_dev = np.array(standard_dev)

np.savez("sho.npz", alphas = alphas, energies = energies, standard_dev = standard_dev)

