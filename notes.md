---
title: "FDTD Parallelization and Perfectly Matched Layers"
author: "Graham Gunlock"
date: "2026-04-22"
---

## Goals and Justification

The major limitation in my previous FDTD simulation was performance and the limited simulation domain. To address this,
instead of using vanilla NumPy I will use Numba. Numba uses JIT compilation to dramatically improve performance, and allows easy parallelization
using a simple function decorator. Since the FDTD has $O(N^3)$ time complexity for a $E_z$ polarized simulation, splitting the electric and magnetic
matrices into components that can be processed in parallel should have immense impact as the resolution increases.

The next major goal of this small-project is to implement the so called uniaxial perfectly matched layer (PML). In my previous simulation
the spatial domain used a simple Dirichlet boundary with $E_z(t), B_x(t), B_y(t) = 0$, this synthetically reproduces a perfect electrical conductor (PEC) at the walls of the simulation.
A consequence of this is that all electromagnetic waves are perfectly reflected at the boundary. This is not ideal because for simulation of anything useful we want waves to be 
able to travel onto infinity.
PML combats this problem by introducing a lossy medium that attenuates all incoming waves (regardless of incidence angle or frequency), while impedance matching to reduce reflections. This task is
non-trivial.

In addition to these goals, I intend to structure my code for effective use in my final project. This looks like a clean and reusable simulation function, tools for saving
simulation data to disk, and physics analysis tools (energy analysis, FFT analysis, etc.). Proper formulation of materials
based on their experimental properties is a top priority. I hope to include the true physical constants, and implement a time scaling method for simulation playback. 



### Notes/Background ---------------------------------------------------------------------

### Important relationships. Brackets [] denote a tensor. In this context just a matrix.

## Relationships to Polarization field and Magnetization field for linear, non-dispersive, anistropic materials
#
# D = e_0 * E + P
# P = e_0 * [chi_e] * E
#
# B = mu_0 (H + M)
# M = [chi_b] * H

## Electric and magnetic susceptibility tensors ([I] is identity tensor/matrix)
#
# [e_r] = [I] + [chi_e]
# [mu_r] = [I] + [chi_b]

## Permittivity and Permeability tensors 
#
# [e] = e_0 * [e_r]           (e_r is relative permittivity)
# [mu] = mu_0 * [mu_r]        (mu_r is relative permeability)

## Constitutive relations: All our materials are diagonalized tensors. For this reason D and B need not be stored themselves.
# D = [e] E
# B = [mu] H

# Induced Current
# J = sigma E     (Ohmic material)

## Maxwells Equations (Time domain) Note: Yee lattice enforces the divergence equations, only need curl equations.
#
# curl(H) = J + d/dt ([e] * E) )                                                        Maxwell-Ampere: recall [e] * E = D
# curl(H) = sigma E + d/dt ([e] * E)                                                    With ohmic conductor
#
# curl(E) = -d/dt ([mu] * H)                                                            Mawxell-Faraday: recall [mu] * H = B

## Maxwells Equations (Frequency domain) Note: for waves of form e^iwt, using a positive convention
#
# curl(H) = J + i*omega * ([e] * E)                                                       Maxwell-Ampere: time-derivatives --> i*omega
# curl(H) = sigma E + i*omega * ([e] * E)                                                 With ohmic conductor
# curl(H) = i*omega * ( [e] + (sigma/i*omega) ) * E                                       Dielectric permittivity and conductivity can be combined into one new term. e', the complex permittivity
# curl(H) = i*omega * (e') * E                                                            e' = [e] + (sigma/i*omega) = [e] - i (sigma/omega)
#
# curl(E) = -i*omega ([mu](omega) * H(omega))                                             Mawxell-Faraday

### -------------------------------------------------------------------------------------



