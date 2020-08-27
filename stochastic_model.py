import cmath
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as bessel
import QPSK_generator
import plotly.graph_objs as gph
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly
import datetime
import QAM_generator


pio.renderers.default = "browser"

def rho(x):
    return bessel.j0(2 * np.pi * x)  # Bessel function 1st kind, 0th order


def hd(dx, h):
    # e = 0.2 * rand(1)[0] + 0.9  # innovation factor
    return rho(dx) * h + (np.sqrt(1 - (abs(rho(dx))) ** 2)) #* e)


def stats(ampl, resolution, UE_x, UE_y):
    UE = ampl[UE_y][UE_x]
    amplis = [20 * np.log10(abs(i / UE)) for i in ampl]
    '''bubble'''
    values_edge_bubble = []
    dist_by_angle = []
    for i in range(0, 360):
        angle = i * np.pi / 180
        x_step = np.cos(angle)
        y_step = np.sin(angle)
        ix = UE_x
        ij = UE_y
        # print(ix, ij)
        inter = amplis[ij][ix]
        prev_val = inter
        dist = 0
        while prev_val >= inter:
            prev_val = inter
            ix += x_step
            ij += y_step
            if ix % 1 != 0 and ij % 1 != 0:
                x1 = int(ix)
                x2 = x1 + 1
                y1 = int(ij)
                y2 = y1 + 1
                dx = abs(ix - x1)
                inter1 = amplis[y1][x1] * (1 - dx) + dx * amplis[y1][x2]
                inter2 = amplis[y2][x1] * (1 - dx) + dx * amplis[y2][x2]
                dy = abs(ij - y1)
                inter = inter1 * (1 - dy) + inter2 * dy
            elif ix % 1 == 0 and ij % 1 != 0:
                y1 = int(ij)
                y2 = y1 + 1
                dy = abs(ij - y1)
                inter = amplis[y1][int(ix)] * (1 - dy) + amplis[y2][int(ix)] * dy
            elif ij % 1 == 0 and ix % 1 != 0:
                x1 = int(ix)
                x2 = x1 + 1
                dx = abs(ix - x1)
                inter = amplis[int(ij)][x1] * (1 - dx) + dx * amplis[int(ij)][x2]
            else:
                inter = amplis[int(ij)][int(ix)]
            dist += resolution
        dist -= resolution
        dist_by_angle.append(dist)
        values_edge_bubble.append(prev_val)
    return [np.var(dist_by_angle), np.average(dist_by_angle), np.var(values_edge_bubble), np.average(values_edge_bubble)]


M = 128
dx = 0.5
a = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1])
s = QAM_generator.create_QAM_signal(a, 4)
noise = np.random.normal(0, np.sqrt(0.5), len(s)) + 1j * np.random.normal(0, np.sqrt(0.5), len(s))
h = np.random.normal(0, np.sqrt(0.5), M) + 1j * np.random.normal(0, np.sqrt(0.5), M)
wH = h.conj().T
aantal_scatterers = 0
dist_between_BS_ant = 0.5
resolution = 0.1
titel = '--M=' + str(M) + '_scatt=' + str(aantal_scatterers) + '_dist-BS-antennas=' + str(
    dist_between_BS_ant) + '_resolution=' + str(resolution) + '_noise-v__zonder_min_max'
header = '(' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") + ')' + titel


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
y = [item * resolution / 10 for item in y2]
x = [item * resolution / 10 for item in x2]

ampl = np.zeros(shape=(len(x), len(y)))
distances = np.zeros(shape=(len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        dx = np.sqrt(x[i] ** 2 + y[j] ** 2)
        hdx = hd(dx, h)
        desired_signal = np.matmul(wH, hdx) / np.sqrt(M)
        ampl[i, j] = desired_signal
        distances[i, j] = dx
    print(i)
# fig = px.imshow(ampl, color_continuous_scale=px.colors.diverging.Portland)
UE_x = int(len(ampl) / 2)
UE_y = int(len(ampl[0]) / 2)
# grootte_vars, grootte_gem, diepte_vars, diepte_gem = stats(ampl,resolution, UE_x, UE_y)
UE = ampl[UE_y][UE_x]
amplis = [20 * np.log10(abs(i / UE)) for i in ampl]
ma = max([max(l) for l in amplis])
mi = -5
fig = go.Figure(data=go.Heatmap(
    z = amplis,
    x = x,
    y = y,
    colorscale='Jet',
    zmin=-30,
    zmax=ma
))
# fig.add_trace(go.Scatter(
#     x=x, y=y,
#     name='antenna',
#     mode='markers',
#     marker_color='red'
# ))
fig.update_layout(
    autosize=False,
    width=850,
    height=800,
    title_text=header,
    font = dict(size = 10)
)
plotly.offline.plot(fig,
                    filename='./result plots/stoc---' + '(' + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M") + ')' + titel + '.html')
fig.show()



fig = go.Figure(data=go.Surface(
    z=amplis,
    x=x,
    y=y,
    colorscale='Jet'
))
fig.update_layout(autosize=False,
                  width=850,
                  height=800,
                  title=header,
                  font=dict(size=10))

plotly.offline.plot(fig,
                    filename='./result plots/' + '(' + datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H-%M") + ')-geomodel-Surface-' + titel + '.html')
fig.show()


# Bins = plt.hist(diepte_gems,10)
# figtitle = './result plots/' + '(' + datetime.datetime.now().strftime(
#                         "%Y-%m-%d_%H-%M") + ')-geo_model--M=' + str(M) + '_resolution=' + str(
#                         resolution) + '.png'
# plt.ylabel("incidents")
# plt.xlabel("difference in power compared to the center")
# plt.title(M)
# plt.savefig(figtitle)
# plt.show()

# jos = np.array([diepte_gems[i] for i in np.nonzero(diepte_gems)[0]])
# jos_vars = np.array([diepte_varsies[i] for i in np.nonzero(diepte_gems)[0]])
# print(np.average(diepte_gems), np.average(jos), np.var(diepte_gems), np.var(jos_vars), np.average(diepte_varsies), np.average(jos_vars), np.var(diepte_varsies), np.var(jos))

