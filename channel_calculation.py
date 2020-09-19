import cmath
from numpy.random import rand
import numpy as np
import scipy.special as bessel
import QPSK_generator
import matplotlib.pyplot as plt


def rho(x):
    return bessel.j0(2 * np.pi * x)  # Bessel function 1st kind, 0th order


def hd(dx, h):
    e = 0.2 * rand(1)[0] + 0.9  # innovation factor
    return rho(dx) * h + (np.sqrt(1 - (abs(rho(dx))) ** 2) * e)


def calc_signal(signal_s, nr_antennas_M, hdx=None, distance_dx=None, weights_hermitian_wH=None, calculation=None, channel_h=None, noise=None):
    if hdx is None:
        hdx = hd(distance_dx, channel_h)        # channel if you move dx from the first channel
    if weights_hermitian_wH is None:            # if weigth is not given: calculate
        wH = channel_h.conj().T
        kanaal = np.matmul(wH, hdx) / nr_antennas_M     # calculate channel weight
    else:
        kanaal = np.matmul(weights_hermitian_wH, hdx) / nr_antennas_M
    yr = kanaal * signal_s                      # received signal
    if noise is not None:
        yr += noise / nr_antennas_M
    sr = [cmath.polar(i) for i in yr]           # in polar coordinates
    if calculation == 'valdB':                  # if value is val: calculate ratio Tx amplitude to Rx amplitude
        amplitudes = [item[0] if int(item[1]) == 0 else -item[0] for item in sr]
        ratio_out_vs_rt_ampl = amplitudes[int(len(amplitudes)/2)+1]/signal_s[int(len(signal_s)/2)+1]
        if ratio_out_vs_rt_ampl < 0:
            ratio_dB = -10
        else:
            ratio_dB = 20*np.log10(ratio_out_vs_rt_ampl)
        return ratio_dB
    elif calculation == 'p':                    # if value is p: calculate phase
        phases = [item[1] for item in sr]
        return phases
    elif calculation == 'val':                  # if value is val: calculate ratio Tx amplitude to Rx amplitude
        amplitudes = [item[0] if int(item[1]) == 0 else -item[0] for item in sr]
        ratio_out_vs_rt_ampl = amplitudes[int(len(amplitudes)/2)+1]/signal_s[int(len(signal_s)/2)+1]
        return ratio_out_vs_rt_ampl
    elif calculation == 'a':                      # if value is a: calculate amplitude
        amplitudes = [item[0] if int(item[1]) == 0 else -item[0] for item in sr]
        return amplitudes
    else:                                       # if value is not given: calculate phase
        amplitudes = [item[0] if int(item[1]) == 0 else -item[0] for item in sr]
        phases = [item[1] for item in sr]
        ratio_out_vs_rt_ampl = amplitudes[int(len(amplitudes)/2)+1] / signal_s[int(len(signal_s)/2)+1]
        return amplitudes, phases, ratio_out_vs_rt_ampl


'''testing area'''
# M = 10000                # amount of antennas in BS
# dx = 0.01                # distance movement of UE
# x = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
# s = QPSK_generator.create_QPSK_signal(x)            # signal TX
# h = np.random.normal(0, np.sqrt(0.5), M) + 1j * np.random.normal(0, np.sqrt(0.5), M)        # channel
# wH = h.conj().T #/ np.linalg.norm(h)         # weights for MR-precoding method
#
# hdx = hd(dx, h)
# amplitudes = calc_signal(s, M, hdx=hdx, weights_hermitian_wH=wH, calculation='a')     # calculate amplitude
# # phases = calc_signal(h, dx, s, M, weights_hermitian_wH=wH, calculation='p')         # calculate phase
# val = calc_signal(s, M, hdx=hdx, weights_hermitian_wH=wH, calculation='val')          # calculate ratio Tx amplitude to Rx amplitude
#
# print(val)
# plt.plot(s.real)
# plt.plot(amplitudes)
# plt.show()
#
# plt.plot(phases)
# plt.show()
