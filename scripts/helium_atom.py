"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020

Final project for PHY 407: Computational Physics.  Instr. Nicolas Grisouard.

This script uses the variational method to solve the 3D Hydrogen molecule

"""

# imports
import numpy as np
import random
import matplotlib.pyplot as plt

# get the problem class from vmc.py
from vmc import problem

# set number of monte carlo steps
N = 500
# create an instance of the "problem" class from vmc.py.  This means we need
# to give it a step width for the derivatives, a step width for the markov
# chain.
helium = problem(0.001, 4, N, 1.5)

# helper function to compute norm of a vector
def norm(r):
    return np.sqrt(np.sum(r**2))

# The potential.  It depends on the alpha parameters this time
def V(alpha, R):
    d = alpha[0]
    R1 = np.array([0, 0, d / 2])
    R2 = np.array([0, 0, -d / 2])

    r1 = R[:, 0]
    r2 = R[:, 1]

    # contributions defined in the report text
    return - 1 / norm(r1 - R1) - 1 / norm(r1 - R2) - 1 / norm(r2 - R1)- 1 / norm(r2 - R2) + 1 / norm(r1 - r2) + 1 / d

# This is our variational "ansatz".  This time, the alpha variable is just a
# scalar.  Its form is defined in the text
def psi(Alpha, R):
    d = Alpha[0]
    alpha = Alpha[1]

    R1 = np.array([0,0,d/2])
    R2 = np.array([0,0,-d/2])

    r1 = R[:,0]
    r2 = R[:,1]


    return (np.exp(-alpha*norm(r1 - R1)) + np.exp(-alpha*norm(r1 - R2))) * (np.exp(-alpha*norm(r2 - R1)) + np.exp(-alpha*norm(r2 - R2)))

# tell the instance of the "problem" class what its ansatz is, and what the
# potential is
helium.psi = psi
helium.V = V

### Main Loop -----------------------------------------------------------------
# define a storage list for the energies
# define a grid with a certain resolution of the two variational parameters
resolution = 50
alphas = np.linspace(0.5, 1.5, resolution)
ds = np.linspace(1.2, 1.6, resolution)
# go through each value of alpha

energies = np.zeros((resolution, resolution))
standard_dev = np.zeros((resolution, resolution))

# loop over both variational parameters
for i in range(resolution):
    for j in range(resolution):
        d, alpha = ds[i], alphas[j]
        helium.alpha = np.array([d, alpha])

        # initialize the location of the electrons
        R = np.array([np.array([0, 0.2, d / 4]),np.array([0,0,-d / 4])]).T
        # tell the problem object which alpha we are working on

        # create a storage array for the energy value at each markov chain point
        estimates = np.array([])

        # log progress
        print("Working on alpha = {}, {}".format(i, j))

        # loop over Monte Carlo Steps
        for _ in range(N):

            #compute monte carlo step
            R = helium.monte_carlo_step(R)

            # calculate local energy
            EL = helium.EL(R)

            # add result to the storage array
            estimates = np.append(estimates, EL)

        # compute mean energy and standard deviation.
        energies[i,j] = np.mean(estimates)
        standard_dev[i,j] = np.std(estimates)

# write data to file for analysis
np.savez("hydrogen_molecule.npz", alphas = alphas, ds = ds, energies = energies, standard_dev = standard_dev)
