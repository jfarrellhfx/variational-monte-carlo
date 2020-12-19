import matplotlib.pyplot as plt
import numpy as np

# Matplotlib Parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)


data = np.load("hydrogen_atom.npz")
alphas = data["alphas"]
energies = data["energies"]
standard_dev = data["standard_dev"]


plt.figure(figsize = (5, 4))
plt.plot(alphas, energies, color = "violet")
plt.fill_between(alphas, energies-standard_dev, energies + standard_dev, color = "thistle")
plt.xlabel("$\\alpha$")
plt.ylabel("$\\langle E_L$")
plt.ylim(-0.8, 0)
plt.tight_layout()
plt.savefig("hydrogen_atom.pdf")
plt.show()


energy = np.min(energies)
error = standard_dev[np.where(np.min(energies) ==  energies)[0][0]]
print("Minimum Energy: {:.3f} +/- {:.3f}".format(energy, error))