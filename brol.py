import cmath
from math import atan
from numpy.random import rand
import numpy as np
import scipy.signal as signal
import random
import matplotlib.pyplot as plt
import scipy.special as bessel
import functions
import QPSK_generator
import plotly.express as px
import plotly.io as pio
import channel_calculation
import angular_distribution as ad
import plotly.graph_objects as go
import datetime
import plotly
import QAM_generator
from skimage import io
from plotly.subplots import make_subplots

pio.renderers.default = "browser"

'''definitions'''
M = 16
message_bits = np.array(
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1])
signal_wave = QAM_generator.create_QAM_signal(message_bits, 64)
# noise v:
# noise = np.random.normal(0, np.sqrt(0.5), (M,len(signal_wave))) + 1j * np.random.normal(0, np.sqrt(0.5), (M,len(signal_wave)))
# noise x:
noise = None
S = 80
dM = 0.5
resolution = 2
titel = '--M=' + str(M) + '_S=' + str(S) + '_dM=' + str(dM) + '_K=' + str(1) + '_resolution=' + str(
    resolution) + '_noise-x__pathloss_random_fase'
header = '(' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") + ')' + titel
print(header)

# place antennas (line @ x=0)
y_setup = list(range(40000 - int(dM * 100 * M / 2), 40000 - int(dM * 100 * M / 2) + int(dM * 100 * M), int(dM * 100)))
y_setup = [i * 0.010 for i in y_setup]
x_setup = list([0 for i in y_setup])
# # random scatt setup
# # place scatterers (random)
# for i in range(S):
#     x_setup.append(random.randint(100000, 700000) / 1000)
#     y_setup.append(random.randint(0, 700000) / 1000)
# # place UE
# x_setup.append(450.3)
# y_setup.append(400.6)
# random_phases = np.random.random(S)
# random_phases = [i*2*np.pi for i in random_phases]
# random_phases = [i*2*np.pi for i in random_phases]

for i in range(S):
    x_setup.append(random.randint(100000, 800000) / 1000)
    y_setup.append(random.randint(0, 800000) / 1000)
random_phases = np.random.random(S) * 2 * np.pi

'''fixed UE setup'''
#    MO     RB       LB      LO       RO      RM      MM       MB      LM
plaatsen_x = [362.319, 595.7, 157.456, 75.149, 635.127, 545.75, 354.62, 365.751, 60.452]
plaatsen_y = [177.642, 559.248, 607.732, 49.53, 99.452, 390.91, 403.956, 617.03, 396.547]
plaatsnaam = ['MO', 'RB', 'LB', 'LO', 'RO', 'RM', 'MM', 'MB', 'LM']
plek = -3
x_setup.append(plaatsen_x[plek])
y_setup.append(plaatsen_y[plek])
# x_setup = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 260.655, 678.968, 354.62]
# y_setup = [395.0, 395.5, 396.0, 396.5, 397.0, 397.5, 398.0, 398.5, 399.0, 399.5, 400.0, 400.5, 401.0, 401.5, 402.0,
#            402.5, 403.0, 403.5, 404.0, 404.5, 563.975, 52.518, 403.956]
# random_phases = [2.91284175, 2.29549205]

'''fixed UE setup'''
# x_setup.extend([184.75, 362.147, 478.3])
# y_setup.extend([148.16, 568.29, 400.6])

'''plotting setup...'''
# plt.scatter(x_setup[:-1], y_setup[:-1])
# plt.scatter(x_setup[-1], y_setup[-1], color='red')
# plt.title(header)
# plt.show()

# calculate distances (a: BS -> scatt; b: scatt -> UE)
distances_a = []  # staat vast, dus moet niet op voorhand gegenereerd worden
for m in range(M):
    distances_a_i = []
    for s in range(S):
        z = s + 2
        distances_a_i.append(np.sqrt((x_setup[m] - x_setup[-z]) ** 2 + (y_setup[m] - y_setup[-z]) ** 2))
    distances_a.append(distances_a_i)
# calc channel vector for UE @original place
h = []
losses = []
for m in range(M):
    contribution = []
    loss = []
    for s in range(S):
        dist_b = np.sqrt((x_setup[-1] - x_setup[-(s + 2)]) ** 2 + (y_setup[-1] - y_setup[-(s + 2)]) ** 2)
        dist = distances_a[m][s] + dist_b
        pathloss = 1 / (16 * np.pi ** 2 * distances_a[m][s] * dist_b)
        loss.append(pathloss)
        phaseshift = 2 * np.pi * ((dist % 1))
        contribution.append(pathloss * np.exp(1j * phaseshift))
    h.append(sum(contribution))
    losses.append(loss)
h = np.array(h)
wH = h.conj().T

'''raster'''
y1 = list(range(-100, 105, 10))
x1 = list(range(-100, 105, 10))
y2 = list(range(-2000, -100, 10))
x2 = list(range(-2000, -100, 10))
y3 = list(range(110, 2010, 10))
x3 = list(range(110, 2010, 10))
y2.extend(y1)
y2.extend(y3)
x2.extend(x1)
x2.extend(x3)
y = [item * resolution / 10 + y_setup[-1] for item in y2]
x = [item * resolution / 10 + x_setup[-1] for item in x2]
'''centrale plaats antenne: x[int(len(x)/2)], y[int(len(y)/2)]'''
# plt.scatter(x, y)
# plt.scatter(x_setup[-1], y_setup[-1], color='red')
# plt.title(header)
# plt.show()

ampl = np.zeros(shape=(len(x), len(y)))
for a in range(len(x)):
    for b in range(len(y)):
        hdx = []
        for m in range(M):
            contribution = []
            for s in range(S):
                dist_b = np.sqrt((x[b] - x_setup[-(s + 2)]) ** 2 + (y[a] - y_setup[-(s + 2)]) ** 2)
                dist = distances_a[m][s] + dist_b
                pathloss = 1 / (16 * np.pi ** 2 * distances_a[m][s] * dist_b)
                phaseshift = 2 * np.pi * ((dist % 1))
                contribution.append(pathloss * np.exp(1j * phaseshift))
            hdx.append(sum(contribution))
        desired_signal = np.matmul(wH, hdx) / np.sqrt(M)
        #                  * signal_wave
        # sr = [cmath.polar(i) for i in desired_signal]
        # amplitudes = [item[0] if int(item[1]) == 0 else -item[0] for item in sr]
        # ratio_out_vs_rt_ampl = amplitudes[int(len(amplitudes) / 2) + 1] / signal_wave[int(len(signal_wave) / 2) + 1]
        # if ratio_out_vs_rt_ampl < 0:
        #     ampl[a, b] = -270
        # else:
        ampl[a, b] = abs(desired_signal)
    print(a)

# plt.plot(signal_wave.real)
# plt.plot(amplitudes)
# plt.title('--M=' + str(M) + '_S=' + str(S) + '_dM=' + str(dM) + '_K=' + str(1))
# plt.show()
# # fig = px.imshow(z=ampl, x=x, y=y, color_continuous_scale=px.colors.diverging.Portland)
pl = ' (' + plaatsnaam[plek] + ')'

title = 'C:/Users/margo/OneDrive/Documenten/Masterproef/simulation/result plots/' + '(' + datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H-%M") + ')-geo_model--M=' + str(M) + '_S=' + str(S) + '_dM=' + str(dM) + '_K=' + str(1) + '_resolution=' + str(
                        resolution) + '_noise-x__pathloss_random_phase.html'
functions.plotter(ampl, title, header, pl, resolution, S, M, x_setup, y_setup, x, y, all_in=True)
