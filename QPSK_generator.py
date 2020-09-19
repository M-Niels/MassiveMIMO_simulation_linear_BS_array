import sys

import numpy as np
import matplotlib.pyplot as plt
import functions


def create_QPSK_signal(x):
    x = functions.data_to_voltage_lvl(x,4)
    I, Q = functions.IQ(x)

    try:
        if I == None or Q == None:
            Ex = ValueError()
            Ex.strerror = "Odd amount of bits given in the signal, while even expected"
            raise Ex
    except ValueError as e:
        print("ValueError Exception!", e.strerror)
        sys.exit()

    t = np.linspace(-1, len(x)/2 + 1, 1000)
    temp = []
    for i in range(len(Q)):                                 # create every piece of cosine on the right point in time
        u = np.heaviside(t-i, 1) - np.heaviside(t-i-1, 1)   # create block function to window the cosine to the right point in time
        phase = Q[i]*((np.pi/2) - (I[i] * np.pi/4))         # phase according to symbol IQ
        a = np.cos((t+phase)*(2*np.pi))                     # cosine with right phase shift
        temp.append(a*u)                                    # store windowed result to plot later on
    s = 0
    for i in range(len(temp)):                              # append all pieces of cosine after each other
        s = s + temp[i]
    return s

'''test section'''
# x = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
# s = create_QPSK_signal(x)
# plt.plot(s)                                          # plot the result
# plt.show()
