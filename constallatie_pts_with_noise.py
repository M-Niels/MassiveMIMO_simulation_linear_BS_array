from math import atan, tan

import numpy as np
import random
import matplotlib.pyplot as plt

num_symbols = 1000
snr = 15
s = [random.randint(1, 4) for a in range(num_symbols)]
print(s)
A = np.sqrt(2)
qpsk_symbols = np.zeros(len(s))*1j
for i in range(len(s)):
    if s[i] == 1:
        qpsk_symbols[i] = A + 1j * A
    elif s[i] == 2:
        qpsk_symbols[i] = A - 1j * A
    elif s[i] == 3:
        qpsk_symbols[i] = - A + 1j * A
    elif s[i] == 4:
        qpsk_symbols[i] = - A - 1j * A

qpsk_symbols = qpsk_symbols + np.random.normal(0, 0.1, len(qpsk_symbols)) + 1j * np.random.normal(0, 0.1, len(qpsk_symbols))
plt.scatter(qpsk_symbols.real, qpsk_symbols.imag)

'''Introduce phase shift'''
qpsk_symbols_shift = np.zeros(len(qpsk_symbols)) * 1j
for i in range(len(qpsk_symbols)):
    phi = atan(qpsk_symbols[i].imag/qpsk_symbols[i].real)
    a = qpsk_symbols[i].real / np.cos(phi)
    nieuwe_phi = phi + np.pi/4
    Re = a * np.cos(nieuwe_phi)
    Im = tan(nieuwe_phi) * Re
    qpsk_symbols_shift[i] = Re + 1j * Im
plt.scatter(qpsk_symbols_shift.real, qpsk_symbols_shift.imag)

plt.show()
