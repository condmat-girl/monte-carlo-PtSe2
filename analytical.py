# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import j0, j1, y0, y1
# from scipy.integrate import cumulative_trapezoid

# # === Parameters ===
# kF = 1        # Fermi wavevector
# J0_amp = 1.0    # RKKY amplitude prefactor
# r_min = 0.01    # Avoid r=0 singularity
# r_max = 20.0
# num_points = 1000

# # === Radial grid ===
# r = np.linspace(r_min, r_max, num_points)

# # === RKKY interaction function in 2D ===
# def rkky_2d(r, kF, J0_amp):
#     x = kF * r
#     return -J0_amp * (j0(x) * y0(x) + j1(x) * y1(x))

# # === Compute interaction ===
# J_r = rkky_2d(r, kF, J0_amp)

# # === Compute cumulative integral: ∫ J(r) · 2πr dr
# J_integrand = J_r * 2 * np.pi * r
# J_cumulative = cumulative_trapezoid(J_integrand, r, initial=0)

# def func(x):
#     return j1(x)

# res = [y0(ri) for ri in r]

# # print([func(ri) for ri in r])

# print(J_cumulative.shape)

# plt.plot(r, J_cumulative,label='integral')
# plt.plot(r, J_integrand, label='integrand')
# plt.plot(r, J_r, label='RKKY')
# plt.grid()
# plt.xlabel('r')
# plt.legend()
# plt.show()







# # # === Plot J(r) ===
# # plt.figure(figsize=(8, 4))
# # plt.plot(r, J_r, label=r'$J(r)$', color='blue')
# # plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# # plt.xlabel('Distance $r$')
# # plt.ylabel('Interaction $J(r)$')
# # plt.title('2D RKKY Interaction')
# # plt.grid(True)
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

# # # === Plot integrated sum: ∫ J(r)·2πr dr ===
# # plt.figure(figsize=(8, 4))
# # plt.plot(r, J_cumulative, label=r'$\int_a^r J(r^\prime)\, 2\pi r^\prime dr^\prime$', color='green')
# # plt.xlabel('Distance $r$')
# # plt.ylabel('Cumulative Interaction Sum')
# # plt.title('Integrated RKKY Interaction in 2D')
# # plt.grid(True)
# # plt.legend()
# # plt.tight_layout()
# # plt.show()



##########################################################################
################ COMPARISON QUAD AND ANALYTICAL INTEGRAL##################
##########################################################################




# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import j0, j1, y0, y1
# from scipy.integrate import quad

# kF = 1.0
# J0_amp = 1.0
# x_min = 0.01
# x_max = 200.0
# num_points = 10000

# x_vals = np.linspace(x_min, x_max, num_points)

# def analytical_integral(x):
#     term1 = x**2 * j0(x) * y0(x)
#     term2 = x**2 * j1(x) * y1(x)
#     term3 = x * j0(x) * y1(x)
#     return -J0_amp / kF**2 * (term1 + term2 - term3)

# def integrand(x):
#     return -J0_amp / kF**2 * x * (j0(x) * y0(x) + j1(x) * y1(x))

# analytical_vals = analytical_integral(x_vals)
# analytical_vals_shifted = analytical_vals - analytical_vals[0]

# quad_vals = []
# for x_end in x_vals:
#     result, _ = quad(integrand, x_min, x_end)
#     quad_vals.append(result)
# quad_vals = np.array(quad_vals)

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(x_vals, analytical_vals_shifted, label='analytical', lw=2)
# plt.plot(x_vals, quad_vals, '-.', label='quad', lw=2)
# plt.xlabel(r'$x = k_F r$')
# plt.ylabel('Integral Value')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(x_vals, analytical_vals_shifted - quad_vals, label='Analytical - Quad', color='blue')
# plt.xlabel(r'$x = k_F r$')
# plt.ylabel('Difference')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()


# from scipy.stats import linregress

# # Generate extended range for x to simulate your x = 150 to x = 200 case
# x_extended = np.linspace(0.01, 200, 5000)
# analytical_extended = analytical_integral(x_extended)
# analytical_shifted_extended = analytical_extended - analytical_extended[0]

# mask = (x_extended >= 150) & (x_extended <= 200)
# x_fit = x_extended[mask]
# y_fit = analytical_shifted_extended[mask]

# # linear regression
# slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
# fit_line = slope * x_fit + intercept
# print(slope, intercept)

# plt.figure(figsize=(10, 6))
# plt.plot(x_extended, analytical_shifted_extended, label='analytical', lw=2)
# plt.plot(x_fit, fit_line, '--', label=f'fit: y = {slope:.5f}x + {intercept:.2f}', color='red')
# plt.xlabel(r'$x = k_F r$')
# plt.ylabel('Integral')
# plt.title('linear fit from x=150')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# print(analytical_shifted_extended[0],analytical_vals_shifted[0])




############################################################################################
#################DEPENDENCE ASYMPTOTIC ON CONSTANT K_F #####################################
############################################################################################
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1, y0, y1
from scipy.integrate import quad
from scipy.stats import linregress

# start = time.time()

# def analytical_integral(x, kF, J0_amp):
#     term1 = x**2 * j0(x) * y0(x)
#     term2 = x**2 * j1(x) * y1(x)
#     term3 = x * j0(x) * y1(x)
#     return -J0_amp / kF**2 * (term1 + term2 - term3)

# kF_values = np.linspace(0.0029, 1, 50)
# intercepts = []

# x_min = 0.01
# x_max = 200.0
# num_points = 5000
# x_vals = np.linspace(x_min, x_max, num_points)

# # Define fit range
# x_fit_min = 150
# x_fit_max = 200

# for kF in kF_values:
#     analytical_vals = analytical_integral(x_vals, kF, J0_amp=1.0)
#     analytical_vals_shifted = analytical_vals - analytical_vals[0]
    
#     # Select fit region
#     mask = (x_vals >= x_fit_min) & (x_vals <= x_fit_max)
#     x_fit = x_vals[mask]
#     y_fit = analytical_vals_shifted[mask]

#     # Linear fit
#     slope, intercept, _, _, _ = linregress(x_fit, y_fit)
#     intercepts.append(intercept)

# np.savetxt('intersepts',intercepts)
# np.savetxt('kF_values',kF_values)

# finish = time.time()


# print(finish- start)


intercepts = np.loadtxt('intersepts.txt')
kF_values = np.loadtxt('kF_values.txt')

plt.figure(figsize=(5, 4))


# Plot intercept vs kF

plt.plot(kF_values, intercepts, 'o-')
plt.xlabel(r'$k_F$')
plt.ylabel(r'$\int_0^{\infty} r J(r) dr $')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()



##################################################################################3
################ PHASE DIAGRAM ####################################################
#####################################################################################



# # Re-import necessary modules after code execution state reset
# import numpy as np
# import matplotlib.pyplot as plt

# # Constants
# k_B = 1.0  # Boltzmann constant (arbitrary units)
# m = 0.5    # Magnetic moment (arbitrary units)

# # Recreate kF_values and load intercepts (I_fit)
# kF_values = np.loadtxt('kF_values.txt')

# # For demonstration, simulate I_fit as decreasing function (replace with np.loadtxt('intersepts') if needed)
# # Simulate intercepts similar to original result
# I_fit = np.loadtxt('intersepts.txt')

# # Define a range of delta (chemical doping)
# delta_values = np.linspace(0.1, 1.0, 50)

# # Prepare TC phase diagram
# TC = np.zeros((len(delta_values), len(kF_values)))

# # Calculate critical temperature for each (delta, kF) pair
# for i, delta in enumerate(delta_values):
#     for j, I in enumerate(I_fit):
#         TC[i, j] = (delta**2 * m**2 * I) / k_B

# # Plotting the phase diagram
# plt.figure(figsize=(10, 6))
# X, Y = np.meshgrid(kF_values, delta_values)
# contour = plt.contourf(X, Y, TC, levels=100, cmap='plasma')
# plt.colorbar(contour, label=r'$T_C$')
# plt.xlabel(r'$k_F$')
# plt.ylabel(r'$\delta$ ')
# plt.title('Phase Diagram')
# plt.grid(True)
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.show()
