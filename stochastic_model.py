import numpy as np
import matplotlib.pyplot as plt
import scipy.special as bessel
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.graph_objects as go
import plotly
import datetime

pio.renderers.default = "browser"


def rho(x):
    return bessel.j0(2 * np.pi * x)  # Bessel function 1st kind, 0th order


def hd(dx, h, M):
    e = np.random.normal(0, np.sqrt(0.5), M) + 1j * np.random.normal(0, np.sqrt(0.5), M)  # innovation factor
    return rho(dx) * h + (np.sqrt(1 - (abs(rho(dx))) ** 2)) * e


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
    return [np.var(dist_by_angle), np.average(dist_by_angle), np.var(values_edge_bubble),
            np.average(values_edge_bubble)]


'''definitions'''
M = 128  # #antennas
resolution = 0.01  # grid step in wavelengths
titel = '--M=' + str(M) + '_resolution=' + str(resolution)
header = '(' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") + ')' + titel

'''area grid around original pos of UE (from, to, step)'''
y1 = list(range(-2000, 2010, 10))
x1 = list(range(-2000, 2010, 10))
y = [item * resolution / 10 for item in y1]
x = [item * resolution / 10 for item in x1]

'''calc channel vector for UE @original place'''
h = np.random.normal(0, np.sqrt(0.5), M) + 1j * np.random.normal(0, np.sqrt(0.5), M)
wH = h.conj().T

'''calc pwr for every pixel of the grid'''
ampl = np.zeros(shape=(len(x), len(y)))
distances = np.zeros(shape=(len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        dx = np.sqrt(x[i] ** 2 + y[j] ** 2)
        hdx = hd(dx, h, M)
        desired_signal = np.matmul(wH, hdx) / np.sqrt(M)
        ampl[i, j] = abs(desired_signal)
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
values_edge_bubble = []
dist_by_angle = []
for i in range(0, 360):
    angle = i * np.pi / 180
    x_step = np.cos(angle)
    y_step = np.sin(angle)
    ix = UE_x
    ij = UE_y
    inter = amplis[ij][ix]
    prev_val = inter
    dist = 0
    while prev_val >= -3:
        try:
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
        except IndexError as e:
            inter = prev_val + resolution
            dist += resolution
            print(angle)
    dist -= resolution
    dist_by_angle.append(dist)
    values_edge_bubble.append(prev_val)
'''Avg intersection'''
tot = amplis[int(len(x) / 2)][int(len(y) / 2):-1]
for i in range(1, 360):
    angle = i * np.pi / 180
    x_step = np.cos(angle)
    y_step = np.sin(angle)
    ix = int(len(amplis) / 2)
    ij = int(len(amplis) / 2)
    graph = [amplis[ij][ix]]
    for j in range(int(len(amplis) / 2) - 1):
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
            graph.append(inter)
        elif ix % 1 == 0 and ij % 1 != 0:
            y1 = int(ij)
            y2 = y1 + 1
            dy = abs(ij - y1)
            inter = amplis[y1][int(ix)] * (1 - dy) + amplis[y2][int(ix)] * dy
            graph.append(inter)
        elif ij % 1 == 0 and ix % 1 != 0:
            x1 = int(ix)
            x2 = x1 + 1
            dx = abs(ix - x1)
            inter = amplis[int(ij)][x1] * (1 - dx) + dx * amplis[int(ij)][x2]
            graph.append(inter)
        else:
            inter = amplis[int(ij)][int(ix)]
            graph.append(inter)
    tot = tot + graph
tot = tot / 360
w = np.linspace(0, int(len(amplis) / 2) + 0.01, num=1000)
w = w * resolution
fig = make_subplots(rows=2, cols=3, column_widths=[250, 250, 750],
                            specs=[[{"colspan": 2},  None, {"rowspan": 2}],
                                   [{"type": "polar"}, None, None]])
fig.add_trace(go.Scatter(x=w, y=20 * np.log10(abs(bessel.j0(2 * np.pi * w))), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=(np.linspace(0, int(len(amplis) / 2) + 0.01, num=len(tot))) * resolution, y=tot,
                                 showlegend=False), row=1, col=1)
fig.add_trace(go.Scatterpolar(theta=np.linspace(0, 360, 360), r=dist_by_angle, mode='lines', showlegend=False), row=2, col=1)
fig.add_trace(go.Heatmap(
    z=amplis,
    x=x,
    y=y,
    colorscale='Jet',
    zmin=-30,
    zmax=ma
),row=1, col=3)
fig.update_layout(autosize=False,
                          width=1400,
                          height=800,
                          title=header,
                          font=dict(size=10),
                          polar=dict(radialaxis=dict(visible=False)),
                          polar2=dict(radialaxis=dict(visible=False)))
fig.show()

# fig = go.Figure(data=go.Surface(
#     z=amplis,
#     x=x,
#     y=y,
#     colorscale='Jet'
# ))
# fig.update_layout(autosize=False,
#                   width=850,
#                   height=800,
#                   title=header,
#                   font=dict(size=10))
#
# plotly.offline.plot(fig,
#                     filename='./result plots/' + '(' + datetime.datetime.now().strftime(
#                         "%Y-%m-%d_%H-%M") + ')-geomodel-Surface-' + titel + '.html')
# fig.show()

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
