import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import threading
import queue

# References to page numbers or equations are from:
# Understanding the Finite-Difference Time-Domain Method, John B. Schneider, www.eecs.wsu.edu/~schneidj/ufdtd, 2010.

## Physical constants
# Free space permittivity and permeability
mu_0 = np.float32(1)
e_0 = np.float32(1)

# Impedence of free-space
eta_0 = np.sqrt(mu_0/e_0)

# Speed of light in vaccum
c_0 = np.float32(1)/(np.sqrt(mu_0*e_0))

# Backend helper function using Numba JIT, for efficency. Numba for loops are compiled, so are fast. Do not need to use vectorization of numpy arrays, can instead loop explicitly.
@njit(parallel=True)
def _simulate_2D(N_x, N_y, dx, dy, dt, E_z, H_x, H_y, inv_e_zz, inv_mu_xx, inv_mu_yy, Psi_Ex, Psi_Ey, Psi_Hx, Psi_Hy, bx, by, Cx, Cy, kx, ky, ticks, X, Y, t_0, source_func, sigma):
    # The main logic is the same as my previous FDTD implementation, last update line uses ampere/faraday to timestep. The difference is
    # CPML causes the gradient operators to change depending on what frequencies are currently present in the fields (eq 11.93, pg 318).
    
    t = t_0
    for tick in range(ticks):
        # Find the curl of E, and update H
        for i in prange(N_x - 1):
            for j in range(N_y - 1): 
                # Calculate forward difference derivative at current index for Ez w.r.t y
                dEz_dy = (E_z[i, j + 1] - E_z[i, j]) / dy
                # Update Psi, using recursive convolution. (eq 11.132, pg. 324)
                Psi_Hx[i,j] = Cy[j] * dEz_dy + by[j] * Psi_Hx[i,j]
                # Find the derivative in new coordinates, accounting for the convolution. (Example eq: 11.110, pg. 320)
                dEz_dy = dEz_dy / ky[j] + Psi_Hx[i,j]

                # Same for Ez w.r.t x
                dEz_dx = (E_z[i + 1, j] - E_z[i, j]) / dx
                # (eq 11.132, pg. 324) 
                Psi_Hy[i,j] = Cx[i] * dEz_dx + bx[i] * Psi_Hy[i,j] 
                # (eq 11.110, pg. 320)
                dEz_dx = (dEz_dx/kx[i]) + Psi_Hy[i,j]

                # Update H_x, using faraday
                H_x[i,j] -=  inv_mu_xx[i,j] * dt * dEz_dy
                # Update H_y, using faraday 
                H_y[i,j] +=  inv_mu_yy[i,j] * dt * dEz_dx

        # Find the curl of H, and update E
        for i in prange(1, N_x - 1):
            for j in range(1, N_y - 1):
                # Calculate backwards difference derivative at current index for Hy w.r.t x
                dHy_dx = (H_y[i,j] - H_y[i - 1, j]) / dx
                # Update Psi, using recursive convolution. (eq 11.132, pg. 324)
                Psi_Ex[i,j] = Cx[i] * dHy_dx + bx[i] * Psi_Ex[i,j]
                # Find the derivative in new coordinates, accounting for the convolution. (Example eq: 11.110, pg 320)
                dHy_dx = (dHy_dx / kx[i]) + Psi_Ex[i,j]

                # Same for Hx w.r.t y
                dHx_dy = (H_x[i,j] - H_x[i, j - 1]) / dy
# (eq 11.132, pg. 324) 
                Psi_Ey[i,j] = Cy[j] * dHx_dy + by[j] * Psi_Ey[i,j]
                # (eq 11.110, pg. 320)
                dHx_dy = (dHx_dy / ky[j]) + Psi_Ey[i,j]
            
                # Update E_z, using Mawxell-ampere and rescaled curl
                # Sigma term accounts for ohmic conductors, source funce is an ideal driving term
                E_z[i,j] += (inv_e_zz[i, j] * dt * ((dHy_dx - dHx_dy)) + source_func(X[i], Y[j], t)   )
                E_z[i,j] -= inv_e_zz[i, j] * sigma[i,j] * E_z[i,j] * dt

        t += dt

    return


class FDTD_2D:
    # N_x, N_y are array size, width and length are physical size
    # E_z polarized, B_x, B_y
    def __init__(self, N_x, N_y, width, length):
        # Save to members
        self.N_x, self.N_y = N_x, N_y
        self.width, self.length = width, length
        
        # Create a linspace for class user to do calculations on, and capture the step for simulation dx,dy
        self.x, self.dx = np.linspace(0, width, N_x, retstep=True, dtype=np.float32)
        self.y, self.dy = np.linspace(0, length, N_y, retstep=True, dtype=np.float32) 
        self.dt = self.dx/np.sqrt(2)
        self.time = 0

        # Initialize physical vector fields
        self.E_z = np.zeros((N_x, N_y), dtype=np.float32)
        self.H_x = np.zeros((N_x, N_y), dtype=np.float32)
        self.H_y = np.zeros((N_x, N_y), dtype=np.float32)

        ## Create anistroptic material fields (permittivity and permeability) (invert them now because otherwise would have to take 1/e_zz, 1/mu_xx, ... every timestep)
        # Permittivity scalar field
        self.inv_e_zz = np.full((N_x, N_y), 1.0/e_0, dtype=np.float32)
        # Permeability tensor field
        self.inv_mu_xx = np.full((N_x, N_y), 1.0/mu_0, dtype=np.float32)
        self.inv_mu_yy = np.full((N_x, N_y), 1.0/mu_0, dtype=np.float32)

        ## Conductivity field for Ohmic materials
        self.sigma = np.zeros((N_x, N_y), dtype=np.float32)

        ## Prepare CPML, this is the convolutional component left after fourier transforming back to time domain. (eq 11.111, pg 320)
        self.Psi_Ex = np.zeros((N_x, N_y), dtype=np.float32)
        self.Psi_Ey = np.zeros((N_x, N_y), dtype=np.float32)
        self.Psi_Hx = np.zeros((N_x, N_y), dtype=np.float32)
        self.Psi_Hy = np.zeros((N_x, N_y), dtype=np.float32)

        ## Initialize the CPML Constants, convolution reduces to be same as vaccum with these intial values
        self.bx = np.ones(N_x, dtype=np.float32) # formula given: (eq 11.126, pg 323)
        self.by = np.ones(N_y, dtype=np.float32) # formula given: (eq 11.126, pg 323)
        self.Cx = np.zeros(N_x, dtype=np.float32) # formula given: (eq 11.127, pg 323)
        self.Cy = np.zeros(N_y, dtype=np.float32) # formula given: (eq 11.127, pg 323)
        self.kx = np.ones(N_x, dtype=np.float32) # Free parameter that can be varied/tweaked to improve results
        self.ky = np.ones(N_y, dtype=np.float32) # ^^
       
        # For queueing frames to workjer thread 
        self.frame_queue = queue.Queue()
        return 
    
    def create_material(self, e_r, mu_xx_r, mu_yy_r, sigma):
        # Dont wanna divide by zero
        # e_r[e_r == 0] = 1
        # mu_xx_r[mu_xx_r == 0] = 1
        # mu_yy_r[mu_yy_r == 0] = 1
        
        self.inv_e_zz *= 1.0/(e_r)
        self.inv_mu_xx *= 1.0/(mu_xx_r)
        self.inv_mu_yy *= 1.0/(mu_yy_r)

        self.sigma += sigma

        return

    # Populate the constants in the regions we want a PML. This hard coded PML creates a smoothed region padding the simulation's Dirchilet boundary.
    def construct_CPML(self, depth, sigma_max, alpha_max, k_max):
        # https://youtu.be/pAjg_odI_YQ?si=H_PsMd83KGMDr2s_&t=991
        # Video explaning why/ and how PML regions should be smoothed

        # Polynomial smoothing order. Want to ramp up the values smoothly.
        m = 4.0

        # PML depth in terms of cells from cartesian depth
        depth_Nx = depth // self.dx
        depth_Ny = depth // self.dy
        
        # Calculate the convolution parameters for x direction ahead of time
        for i in range(self.N_x):
            depth_ratio = 0.0

            # Check if we're in the left PML 
            if i < depth_Nx:
                # What percent of the way through are we
                depth_ratio = (depth_Nx - i) / depth_Nx
            # Check if we're in the right PML
            elif i >= (self.N_x - depth_Nx):
                # Calculate distance from inner edge of PML
                start = (self.N_x - depth_Nx)
                distance = i - start                
                depth_ratio = (distance + 1) / (depth_Nx) 
    
            if depth_ratio > 0.0:
                # Smooth the the values in the PML so the loss is added gradually and not all at once.
                # A sharp change will cause reflections
                kx_i = 1.0 + (k_max - 1.0) * (depth_ratio ** m)
                sigmax_i = sigma_max * (depth_ratio ** m)
                alphax_i = alpha_max * (1.0 - depth_ratio)

                # Calculate bx[i] from sigma alpha and k. (Equation 11.126, page 323)
                self.bx[i] = np.exp(-1 * ( (alphax_i) + (sigmax_i/(kx_i)) ) * (1/e_0) * self.dt)
                # Calculate cx[i] from sigma alpha bx and k. (Equation 11.127, page 323)
                self.Cx[i] = ( (sigmax_i)/(sigmax_i * kx_i + (kx_i**2) * alphax_i ) ) * (self.bx[i] - 1)

                # Update kx
                self.kx[i] = kx_i

        # Calculate the convolution parameters for y direction ahead of time
        for j in range(self.N_y):
            depth_ratio = 0.0

            # Check if we're in the bottom PML 
            if j < depth_Ny:
                # What percent of the way through are we
                depth_ratio = (depth_Ny - j) / depth_Ny
            # Check if we're in the top PML
            elif j >= (self.N_y - depth_Ny):
                # Calculate distance from inner edge of PML
                start = (self.N_y - depth_Ny)
                distance = j - start 
                depth_ratio = (distance + 1) / (depth_Ny) 
    
            if depth_ratio > 0.0:
                # Same procedure as for x
                ky_j = 1.0 + (k_max - 1.0) * (depth_ratio ** m)
                sigmay_j = sigma_max * (depth_ratio ** m)
                alphay_j = alpha_max * (1.0 - depth_ratio)
                
                # Calculate by[j] from sigma alpha and k. (Equation 11.126, page 323)
                self.by[j] = np.exp(-1 * ( (alphay_j) + (sigmay_j/(ky_j)) ) * (1/e_0) * self.dt)
                # Calculate cy[j] from sigma alpha by and k. (Equation 11.127, page 323)
                self.Cy[j] = ( (sigmay_j)/(sigmay_j * ky_j + (ky_j **2) * alphay_j ) ) * (self.by[j] - 1)

                # Update ky
                self.ky[j] = ky_j

        return
    
    # Queue consumer thread for saving frames. https://docs.python.org/3/library/queue.html
    def _save_frames(self, filename):
        # open file
        with open(f'{filename}.npy', 'wb') as file:
            tick = 0
            while True:
                # Wait for a frame to get added to queue
                frame = self.frame_queue.get()

                # stop when no more frames
                if frame is None: 
                    self.frame_queue.task_done()
                    break

                # Save frame (print for debug)
                print(f"Saving frame: {tick}")
                tick += 1

                # np save stores current arrays in a .npy binary file
                np.save(file, frame)
                
                # Unblock join() in simulate. Simulate will block until worker thread finished.
                self.frame_queue.task_done()
        return

    # Source_func is a @jit callback that gets passed into simulate backend. Must have signature:
    # def source_func(x, y, t) -> float32
    # where x, y, t are float32 NOT arrays
    def simulate(self, source_func, steps, frame_count, filename = "temp"):
        # Spawn worker thread
        threading.Thread(target=self._save_frames, args=(filename,), daemon=True).start()
        
        # steps per frame
        steps_per = steps//frame_count

        # Calling _simulate_2D from Python has overhead, so don't call it every timestep. Call it only once per frame, and tell it to run steps_per times.
        time = 0
        for tick in range(steps//steps_per):
            # Call the back-end parallel simulation
            _simulate_2D(self.N_x, self.N_y, self.dx, self.dy, self.dt, \
                       self.E_z, self.H_x, self.H_y, self.inv_e_zz, self.inv_mu_xx, self.inv_mu_yy, \
                       self.Psi_Ex, self.Psi_Ey, self.Psi_Hx, self.Psi_Hy, self.bx, self.by, self.Cx, \
                       self.Cy, self.kx, self.ky, steps_per, self.x, self.y, time, source_func, self.sigma) 

            # Update time
            time += self.dt * steps_per 

            # Cache the current E_z state, and queue to frame_queue so worker can write to disk
            frame = self.E_z
            self.frame_queue.put(frame)
            print("Frame queued")

        print("simulation complete")
        # Signal thread to stop
        self.frame_queue.put(None)
        # Block until worker finishes 
        self.frame_queue.join()
        return 
    
    def playback(self, filename = "temp", fps=30.0):
        print("playback called")
        # https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load
        plt.ion()
        fig, ax = plt.subplots()
        # Open binary .npy
        with open(f'{filename}.npy', 'rb') as file:
            tick = 0
            while True:
                try:
                    # Grab the saved array from binary file
                    E_z = np.load(file)
                    if tick == 0:
                        # on first frame need to create the plot
                        im = ax.imshow(E_z.T, cmap='berlin', vmin=-0.1, vmax=0.1, origin='lower')
                    else:
                        # update with current frame
                        im.set_data(E_z.T)
                        ax.set_title(f"t = 0, tick={tick}")
                        plt.pause(0.001)
                    tick +=1
                except OSError as error:
                    # Exception thrown by np.load() if file is unreachable/not binary
                    print(f"No .npy file with name: {filename}, {error}")
                    break
                except EOFError:
                    # Reached end of file
                    break
        #plt.close()
        return

