import matplotlib.pyplot as plt
import numpy as np

# Matplotlib Parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)


data = np.load("hydrogen_molecule.npz")
alphas = data["alphas"]
ds = data["ds"]
energies = data["energies"]
standard_dev = data["standard_dev"]

i = np.where(np.min(energies) ==  energies)[0][0]
j = np.where(np.min(energies) ==  energies)[1][0]

plt.figure(figsize = (5, 4))
plt.plot(ds, energies[:, j], color = "black")
plt.fill_between(ds, energies[:,j]-standard_dev[:,j], energies[:,j] + standard_dev[:, j], color = "lightslategray")
plt.xlabel("$\\alpha$")
plt.ylabel("$\\langle E_L $")
#plt.ylim(-0.8, 0)
plt.tight_layout()
plt.savefig("hydrogen_molecule.pdf")
plt.show()


energy = np.min(energies)

error = standard_dev[i,j]
print("Minimum Energy: {:.3f} +/- {:.3f}".format(energy, error))
print("Best d {:.3f}".format(ds[i]))
print("Best alpha {:.3f}".format(alphas[j]))