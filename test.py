import matplotlib.pyplot as plt
from triangle import *

# defines how many atoms will be in row and column
rows, cols = 5, 5
# define the side of triangle
spacing = 1.0
k_f = 1.0
U0 = 1.0
steps = 100000

# test
lattice = TriangularLattice(rows, cols, spacing, k_f, U0)



########################################################################
################## the plotting of the system ##########################
########################################################################



# magnetizations = []
# for T in temperatures:
#     lattice.run_monte_carlo(steps, T)
#     magnetizations.append(lattice.average_magnetization_x())


# plots! the system sanity-check
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(lattice.lattice_points[:, 0], lattice.lattice_points[:, 1], np.zeros(len(lattice.lattice_points)), s=20, color='b')
# ax.quiver(lattice.lattice_points[:, 0], lattice.lattice_points[:, 1], np.zeros(len(lattice.lattice_points)),
#           lattice.magnetic_moments[:, 0], lattice.magnetic_moments[:, 1], lattice.magnetic_moments[:, 2],
#           length=0.5, color='b')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.set_zlim(-2,2)
# ax.set_title("Triangular Lattice with 3D Magnetic Moments")
# plt.show()




########################################################################
########### the magnetization plot of temperature ######################
########################################################################



# temperatures = np.linspace(0.1, 5.0, 20)

# lattice = TriangularLattice(rows, cols, spacing, k_f, U0)

# magnetizations = []
# for T in temperatures:
#     lattice.run_monte_carlo(steps, T)
#     magnetizations.append(lattice.average_magnetization_x())



# # magnetiyation plot
# plt.figure(figsize=(8, 6))
# plt.plot(temperatures, magnetizations, marker='o', linestyle='-', color='b')
# plt.xlabel("Temperature [T]")
# plt.ylabel("m_x")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
