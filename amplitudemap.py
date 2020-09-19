import cmath
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as bessel
import QPSK_generator
import plotly.graph_objs as gph
import plotly.express as px
import plotly.io as pio
import channel_calculation
import plotly.graph_objects as go
import plotly
import datetime

pio.renderers.default = "browser"


def rho(x):
    return bessel.j0(2 * np.pi * x)  # Bessel function 1st kind, 0th order


def hd(dx, h):
    e = 0.2 * rand(1)[0] + 0.9  # innovation factor
    return rho(dx) * h + (np.sqrt(1 - (abs(rho(dx))) ** 2) * e)


M = 100
dx = 0.01
a = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
s = QPSK_generator.create_QPSK_signal(a)
h = np.random.normal(0, np.sqrt(0.5), M) + 1j * np.random.normal(0, np.sqrt(0.5), M)
wH = h.conj().T

y1 = list(range(-100, 105, 10))
x1 = list(range(-100, 105, 10))
y2 = list(range(-500, -100, 10))
x2 = list(range(-500, -100, 10))
y3 = list(range(110, 510, 10))
x3 = list(range(110, 510, 10))
y2.extend(y1)
y2.extend(y3)
x2.extend(x1)
x2.extend(x3)
y = [item/100 for item in y2]
x = [item/100 for item in x2]

ampl = np.zeros(shape=(len(x), len(y)))
distances = np.zeros(shape=(len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        dx = np.sqrt(x[i] ** 2 + y[j] ** 2)
        val = channel_calculation.calc_signal(s, M, weights_hermitian_wH=wH, calculation='val', channel_h=h, distance_dx=dx)
        ampl[i, j] = val
        # hdx = hd(dx, h)
        # en = np.matmul(wH, hdx) / M
        # ampl[i, j] = en
        distances[i, j] = dx

fig = px.imshow(ampl, color_continuous_scale=px.colors.diverging.Portland)

# fig.add_trace(go.Scatter(
#     x=x, y=y,
#     name='antenna',
#     mode='markers',
#     marker_color='red'
# ))

fig.show()

plotly.offline.plot(fig,
                    filename='C:/Users/margo/OneDrive/Documenten/Masterproef/simulation/result plots/amplitudemap---M-' + str(
                        M) + '-#scatt=0 (' + datetime.datetime.now().strftime(
                        "%d-%m-%Y_%H-%M") + ').html')
# fig = px.imshow(distances, color_continuous_scale=px.colors.diverging.Portland)
# fig.show()
