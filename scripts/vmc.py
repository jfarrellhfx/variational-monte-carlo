import numpy as np
import matplotlib.pyplot as plt
import random


class problem:
    """
    This is a general class to hold each problem we are goign to work on, e.g.
    the harmonic oscillator, or the hydrogen atom.  For each problem, we can
    make an instance of this class and give it the corresponding potential and
    trial wave function
    """


    @staticmethod
    def laplacian(f, R, h):
        """
        Compute the laplacian of a function f(R).

        :param f: a function of R
        :param R: array of coorinates (rows give spatial coordinates 1 - 3, and
        columns give particle indices. If we had 2 particles in 2 dimensions,
        it would be a 2 X 2 array (one column for each particle)
        :param h: the step width to use for calculating the numerical second
        derivatives
        :return:
        """
        # we need to do a different thing depending on how many dimensions we
        # are working in
        # in 1D, just use centered difference
        if type(R) == float:
            return (f(R + h) - 2 * f(R) + f(R - h)) / h ** 2

        # in 3d, use centered difference, but need to do it in each dimension
        elif len(R.shape) == 1:
            lap = 0
            for i in range(len(R)):
                H = np.zeros_like(R)
                H[i] = h

                lap += (f(R + H) - 2 * f(R) + f(R-H)) / h **2
            return lap

        # for more than one particle, use centered difference for each coordinate
        # of each particle
        else:
            lap = 0
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    H = np.zeros_like(R)
                    H[i,j] = h
                    lap += (f(R + H) - 2 * f (R) + f(R - H)) / h ** 2
            return lap

    def __init__(self, derivative_step_width, mc_step_width, N, alpha):
        """
        Initialize the instance

        :param derivative_step_width: step width to use for numerical derivatives
        :param mc_step_width: the maximum possible step size for the markov chain
        """

        # initialize
        self.derivative_step_width = derivative_step_width
        self.mc_step_width = mc_step_width

        self.alpha = 0

        # This part is tricky: we want the wave function to depend on alpha,
        # but it's nicer to program it as a function of one variable (so we can
        # use, for instance, the problem.laplacian() method defined above.  To
        # do so, we can define a quick funtion psi_var that is the same as
        # psi, but has the alpha "built in"
        self.psi_var = lambda R: self.psi(self.alpha, R)
        self.V_var = lambda R: self.V(self.alpha, R)


    def psi(self, alpha, R):
        """
        The user is to override this with the trial wave function, a function
        of alpha and R

        :param alpha: variational parameters or array thereof
        :param R: array of coorinates (rows give spatial coordinates 1 - 3, and
        columns give particle indices. If we had 2 particles in 2 dimensions,
        it would be a 2 X 2 array (one column for each particle)
        :return: the wave function
        """
        return np.ones(R.shape)


    def V(self, alpha, R):
        """
        The potential energy of the problem.  The user should supply this

        :param R: array of coorinates (rows give spatial coordinates 1 - 3, and
        columns give particle indices. If we had 2 particles in 2 dimensions,
        it would be a 2 X 2 array (one column for each particle)
        :return: V(R), the potential energy
        """

        return np.zeros(len(R))


    def Hamiltonian(self, R):
        """
        Function that gives the result of operating on the wavefunction psi
        with the hamiltonian operator

        :param R: array of coorinates (rows give spatial coordinates 1 - 3, and
        columns give particle indices. If we had 2 particles in 2 dimensions,
        it would be a 2 X 2 array (one column for each particle)
        :return:
        """
        kinetic = -1 / 2 * problem.laplacian(self.psi_var, R, self.derivative_step_width)
        potential = self.V_var(R) * self.psi_var(R)
        return float(kinetic + potential)


    def EL(self, R):
        """
        Local energy as defined in report

        :param R: coordinates
        :return: local energy operator
        """
        return float(1 / self.psi_var(R) * self.Hamiltonian(R))


    def monte_carlo_step(self, R):
        """
        Calculate a step of the monte carlo markov chain method

        :param R: coordinates
        :return: new coordinates after step
        """

        # get the step width that we are using for our monte carlo calculations
        # (maximum jump).
        h = self.mc_step_width

        # do a different thing for each dimension.  But basically, we are adding a
        # random number between -0.5 * h and 0.5 * h to our step, and
        # computing the probability densities again

        #
        if type(R) == float:

            H = random.random - 0.5
            R_new = R + H
        else:
            # this is a command to make an array of random numbers, very useful
            H = h * (np.random.random_sample(R.shape) - 0.5)
            R_new = np.copy(R + H)

        # compute probability density before and after step
        P1 = np.abs(self.psi_var(R)) ** 2
        P2 = np.abs(self.psi_var(R_new)) ** 2

        # accept step according to Markov Chain probability
        if P2 / P1 >= random.random():
            R = R_new
        else:
            R = R
        return R







