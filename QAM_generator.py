import sys

import numpy as np
import matplotlib.pyplot as plt
import functions

#MO: aantal energieniveau's
def create_QAM_signal(x, n):
    s = functions.data_to_voltage_lvl(x, n)
    I, Q = functions.IQ(s)

    try:
        if I == None or Q == None:
            Ex = ValueError()
            Ex.strerror = "wrong amount of bits given in the signal. N*" + str(
                int(2 * np.log2(np.sqrt(n)))) + " expected with N=integer"
            raise Ex
    except ValueError as e:
        print("ValueError Exception!", e.strerror)
        sys.exit()

    t = np.linspace(-1, len(x) / 2 + 1, 1000)
    temp = []
    for i in range(len(Q)):  # create every piece of cosine on the right point in time
        u = np.heaviside(t - i, 1) - np.heaviside(t - i - 1,
                                                  1)  # create block function to window the cosine to the right point in time
        # phase = Q[i]*((np.pi/2) - (I[i] * np.pi/4))       # phase according to symbol IQ
        a = I[i] * np.cos(t * (2 * np.pi)) - Q[i] * np.sin(t * (2 * np.pi))  # cosine with right phase shift
        temp.append(a * u)  # store windowed result to plot later on
    s = 0
    for i in range(len(temp)):  # append all pieces of cosine after each other
        s = s + temp[i]
    return s[:int((len(x) / (np.log2(np.sqrt(n))*2) + 2) * (1000/(len(x)/2+2)))]


'''Test zone'''
# x = np.array(
#     [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
#      0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1])
# s = create_QAM_signal(x, 1024)
# plt.plot(s)  # plot the result
# plt.show()
