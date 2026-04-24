from FDTD import *
import numpy as np
from numba import njit

# Resolution and dimensions
RESOLUTION = 1000

Nx, Ny = RESOLUTION, RESOLUTION 
width, length = 20.0, 20.0

# Initialize
FDTD = FDTD_2D(Nx, Ny, width, length)

# Parameters: depth, sigma_max, alpha_max, k_max.
# These numbers are just guesses. After fiddling with different values, these gave best results.
depth = width * 0.2
FDTD.construct_CPML(depth, 3.0, 0.0001, 1.0)

# Create a source
@njit
def ramp_source(x, y, t):
        # Oscillator parametrs
        amplitude = 4
        frequency = 1
        period = 1/frequency

        # Delta spike
        pos_x = 5.0
        pos_y = 5.0
        tol = 0.010
        
        # If the x,y coordinates are at position (pos_x, pos_y) +- tolerance
        if (x - tol <= pos_x <= x + tol) and (y - tol <= pos_y <= y + tol):
            # Inject an oscillating current, that is smoothed as it turns on
            oscillator = (amplitude * np.cos(2 * np.pi * frequency * t))
            envelope = (1 - np.exp((-1 * (t))/(3 * period))) # If oscillator is turned on instantly, there is a step function at t=0. This introduces high frequency harmonics that CPML handles poorly.

            return oscillator * envelope
        else:
            # Only inject at delta-spike
            return 0

# Construct a material
X, Y= np.meshgrid(FDTD.x, FDTD.y)

# Circle with radius 3, centered at (10, 10), with relative Permittivity = 6.5
epsilon_r = ((X - 10)**2 + (Y - 10)**2 < 9) * 6.5 
# Must be 1 everywhere else, NOT zero.
epsilon_r[epsilon_r == 0] = 1

mu_r = 1

# Create three circles with conductivity of 70 siemens/meter. Conductivity is zero by default so don't need to switch zeros->ones
sigma = ((X - 3)**2 + (Y - 4)**2 < 1) * 70.0 
sigma += ((X - 15)**2 + (Y - 4)**2 < 1) * 70.0 
sigma += ((X - 12)**2 + (Y - 6)**2 < 1) * 70.0 

FDTD.create_material(epsilon_r, mu_r, mu_r, sigma)

# Display the materials and CPML 

# CPML Constant C
Cx, Cy = np.meshgrid(FDTD.Cx, FDTD.Cy)
C = Cx + Cy

plt.imshow(C, cmap='viridis', origin='lower')
plt.title("CPML")
plt.colorbar() 
plt.show()

# Conductivity
plt.imshow(sigma.T, cmap='viridis', origin='lower')
plt.title("Conductivity")
plt.colorbar() 
plt.show()

# Permittivity
plt.imshow(epsilon_r.T, cmap='viridis', origin='lower')
plt.title("Relative Permittivity")
plt.colorbar() 
plt.show()

FDTD.simulate(ramp_source, 4000, 960)
FDTD.playback()

