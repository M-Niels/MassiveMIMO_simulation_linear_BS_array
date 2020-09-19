import cmath
from math import atan
from numpy.random import rand
import numpy as np
import scipy.special as bessel
import matplotlib.pyplot as plt
import QPSK_generator

M = 100
dx = 0


def rho(x):
    return bessel.j0(2 * np.pi * x)  # Bessel function 1st kind, 0th order


def hd(dx, h):
    e = 0.2 * rand(1)[0] + 0.9  # innovation factor
    return rho(dx) * h + (np.sqrt(1 - (abs(rho(dx))) ** 2) * e)


h = np.random.normal(0, 0.5, M) + 1j * np.random.normal(0, 0.5, M)

x = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
s = QPSK_generator.create_QPSK_signal(x)
phases_per_twintigste_van_een_wavelength = []
plek_per_twintigste_van_een_wavelength = []
for i in range(201):
    hdx = hd(dx, h)
    hdxH = hdx.conj().T
    kanaal = np.matmul(hdxH, h) / M
    # print("%.2f" % dx, atan(kanaal.imag / kanaal.real))
    phase = atan(kanaal.imag / kanaal.real)
    # if phase < 0:
    #     phase += np.pi
    phases_per_twintigste_van_een_wavelength.append(phase)
    plek_per_twintigste_van_een_wavelength.append(dx)
    dx += 0.05
plt.plot(plek_per_twintigste_van_een_wavelength, phases_per_twintigste_van_een_wavelength)
plt.show()
