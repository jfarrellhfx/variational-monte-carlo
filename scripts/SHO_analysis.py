import matplotlib.pyplot as plt
import numpy as np

# Matplotlib Parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)


data = np.load("sho.npz")
alphas = data["alphas"]
energies = data["energies"]
standard_dev = data["standard_dev"]


plt.figure(figsize = (5, 4))
plt.plot(alphas, energies, color = "steelblue")
plt.fill_between(alphas, energies-standard_dev, energies + standard_dev, color = "lightsteelblue")
plt.xlabel("$\\alpha$")
plt.ylabel("$\\langle E_L / \\hbar \\omega \\rangle$")
plt.tight_layout()
plt.savefig("sho.pdf")
plt.show()

energy = np.min(energies)
error = standard_dev[np.where(np.min(energies) ==  energies)[0][0]]
print("Minimum Energy: {:.3f} +/- {:.3f}".format(energy, error))