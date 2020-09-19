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
import plotly.graph_objects as go
import angular_distribution as ad
import datetime
import plotly
import QAM_generator
from skimage import io
from plotly.subplots import make_subplots

pio.renderers.default = "browser"

'''definitions'''
M = 128
message_bits = np.array(
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1])
signal_wave = QAM_generator.create_QAM_signal(message_bits, 64)
# noise v:
# noise = np.random.normal(0, np.sqrt(0.5), (M,len(signal_wave))) + 1j * np.random.normal(0, np.sqrt(0.5), (M,len(signal_wave)))
# noise x:
noise = None
S = 200
dM = 0.5
resolution = 0.1
aantal_runs = 4
titel = '--M=' + str(M) + '_S=' + str(S) + '_dM=' + str(dM) + '_resolution=' + str(resolution) + '_R=' + str(aantal_runs) + '_noise-x__pathloss_gn_random_fase'
header = '(' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") + ')' + titel
print(header)



'''fixed UE setup'''
            #    MO     RB       LB      LO       RO      RM      MM       MB      LM
plaatsen_x = [362.319, 595.7, 157.456, 75.149, 635.127, 545.75, 354.62, 365.751, 60.452]
plaatsen_y = [177.642, 559.248, 607.732, 49.53, 99.452, 390.91, 403.956, 617.03, 396.547]
plaatsnaam = ['MO', 'RB', 'LB', 'LO', 'RO', 'RM', 'MM', 'MB', 'LM']
plek = -3
pl = ' (' + plaatsnaam[plek] + ')'

'''raster'''
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
y = [item * resolution / 10 + plaatsen_y[plek] for item in y2]
x = [item * resolution / 10 + plaatsen_x[plek] for item in x2]

x_setup = []
y_setup = []
gem_ampli = np.zeros(shape=(len(x), len(y)))
for r in range(aantal_runs):
    # place antennas (line @ x=0)
    y_setup = list(
        range(40000 - int(dM * 100 * M / 2), 40000 - int(dM * 100 * M / 2) + int(dM * 100 * M), int(dM * 100)))
    y_setup = [i * 0.010 for i in y_setup]
    x_setup = list([0 for i in y_setup])
    '''random scatt setup'''
    # place scatterers (random)
    for i in range(S):
        x_setup.append(random.randint(100000, 700000) / 1000)
        y_setup.append(random.randint(0, 700000) / 1000)
    random_phases = np.random.random(S)*2*np.pi
    x_setup.append(plaatsen_x[plek])
    y_setup.append(plaatsen_y[plek])
    
    # '''plotting setup...'''
    # plt.scatter(x_setup[:-1], y_setup[:-1])
    # plt.scatter(x_setup[-1], y_setup[-1], color='red')
    # plt.title('(' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") + ')'+'-M=' + str(M) + '_S=' + str(S) + '_resolution=' + str(resolution) + '_R=' + str(aantal_runs))
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
    for m in range(M):
        contribution = []
        for s in range(S):
            dist_b = np.sqrt((x_setup[-1] - x_setup[-(s + 2)]) ** 2 + (y_setup[-1] - y_setup[-(s + 2)]) ** 2)
            dist = distances_a[m][s] + dist_b
            pathloss = 1 / (16 * np.pi**2 * dist_b * distances_a[m][s])
            phaseshift = 2 * np.pi * ((dist % 1) + random_phases[s])
            contribution.append(pathloss * np.exp(1j * phaseshift))
        h.append(sum(contribution))
    h = np.array(h)
    wH = h.conj().T
    
    ampl = np.zeros(shape=(len(x), len(y)))
    for a in range(len(y)):
        for b in range(len(x)):
            hdx = []
            for m in range(M):
                contribution = []
                for s in range(S):
                    dist_b = np.sqrt((x[b] - x_setup[-(s + 2)]) ** 2 + (y[a] - y_setup[-(s + 2)]) ** 2)
                    dist = distances_a[m][s] + dist_b
                    pathloss = 1 / (16 * np.pi**2 * dist_b * distances_a[m][s])
                    phaseshift = 2 * np.pi * ((dist % 1) + random_phases[s])
                    contribution.append(pathloss * np.exp(1j * phaseshift))
                hdx.append(sum(contribution))
            desired_signal = np.matmul(wH, hdx) / np.sqrt(M)
            ampl[a, b] = abs(desired_signal)
        print(str(r)+'-'+str(a))
    title ='./result plots/' + '(' + datetime.datetime.now().strftime(
                            "%Y-%m-%d_%H-%M") + ')-geo_model_gem--' + str(r+1) + titel + '.html'
    functions.plotter(ampl, title, header, pl, resolution, S, M, x_setup, y_setup, x, y, all_in=True)
 
    gem_ampli += ampl/aantal_runs
    title = './result plots/' + '(' + datetime.datetime.now().strftime(
                            "%Y-%m-%d_%H-%M") + ')-geo_model_gem--g' + str(r+1) + titel + '.html'
    functions.plotter(gem_ampli, title, header, pl, resolution, S, M, x_setup, y_setup, x, y, all_in=False)
    
gem_ampli = gem_ampli

title = './result plots/' + '(' + datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H-%M") + ')-geo_model_gem--' + titel + '.html'
functions.plotter(gem_ampli, title, header, pl, resolution, S, M, x_setup, y_setup, x, y, all_in=False)
