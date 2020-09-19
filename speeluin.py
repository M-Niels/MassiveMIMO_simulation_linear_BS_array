import cmath
from math import atan
from numpy.random import rand
import numpy as np
import scipy.signal as signal
import random
import matplotlib.pyplot as plt
import scipy.special as bessel
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

'''plotter voor gemiddelde'''
def gem_plotter(ampl, file_name, fig_header, pl, resolution, S, M, x_setup, y_setup, x, y, wH=None, all_in=True, mini=None,
            maxi=None):
    UE = ampl[int(len(ampl) / 2)][int(len(ampl[0]) / 2)]
    amplis = [20 * np.log10(i / UE) for i in ampl]
    if mini is not None and maxi is None:
        maxi = max([max(i) for i in ampl])
    elif maxi is not None and mini is None:
        mini = min([min(i) for i in ampl])
    if all_in:
        '''Gem doorsnede'''
        tot = amplis[int(len(x) / 2)][int(len(y) / 2):]
        for i in range(1, 360):
            hoek = i
            x_step = np.cos(hoek)
            y_step = np.sin(hoek)
            ix = int(len(amplis) / 2)
            ij = int(len(amplis) / 2)
            grafiek = [amplis[ix][ij]]
            for j in range(int(len(amplis) / 2)):
                ix += x_step
                ij += y_step
                if ix % 1 != 0 and ij % 1 != 0:
                    x1 = int(ix)
                    x2 = x1 + 1
                    y1 = int(ij)
                    y2 = y1 + 1
                    dx = abs(ix - x1)
                    inter1 = amplis[x1][y1] * (1 - dx) + dx * amplis[x2][y1]
                    inter2 = amplis[x1][y2] * (1 - dx) + dx * amplis[x2][y2]
                    dy = abs(ij - y1)
                    inter = inter1 * (1 - dy) + inter2 * dy
                    grafiek.append(inter)
                elif ix % 1 == 0:
                    y1 = int(ij)
                    y2 = y1 + 1
                    dy = abs(ij - y1)
                    inter = amplis[int(ix)][y1] * (1 - dy) + amplis[int(ix)][y2] * dy
                    grafiek.append(inter)
                elif ij % 1 == 0:
                    x1 = int(ix)
                    x2 = x1 + 1
                    dx = abs(ix - x1)
                    inter = amplis[x1][int(ij)] * (1 - dx) + dx * amplis[x2][int(ij)]
                    grafiek.append(inter)
            tot = tot + grafiek
        tot = tot / 360
        w = np.linspace(0, int(len(amplis) / 2) + 0.01, num=1000)
        w = w * resolution

        '''bubble'''
        afstand_per_hoek = []
        for i in range(0, 360):
            hoek = i * np.pi / 180
            x_step = np.cos(hoek)
            # print(x_step)
            y_step = np.sin(hoek)
            ix = int(len(amplis) / 2)
            ij = int(len(amplis) / 2)
            # print(ix, ij)
            inter = [amplis[ij][ix]]
            vorige_waarde = inter
            afstand = 0
            while vorige_waarde >= inter:
                vorige_waarde = inter
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
                    # print('1')
                elif ix % 1 == 0 and ij % 1 != 0:
                    y1 = int(ij)
                    y2 = y1 + 1
                    dy = abs(ij - y1)
                    inter = amplis[y1][int(ix)] * (1 - dy) + amplis[y2][int(ix)] * dy
                    # print('2')
                elif ij % 1 == 0 and ix % 1 != 0:
                    x1 = int(ix)
                    x2 = x1 + 1
                    dx = abs(ix - x1)
                    inter = amplis[int(ij)][x1] * (1 - dx) + dx * amplis[int(ij)][x2]
                    # print('3')
                else:
                    inter = amplis[int(ij)][int(ix)]
                #     print('4')
                # print(inter)
                afstand += resolution
            afstand -= resolution
            afstand_per_hoek.append(afstand)

        fig = make_subplots(rows=2, cols=2, column_widths=[700, 850],
                            specs=[[None, {"rowspan": 2}],
                                   [{"type": "polar"}, None]]
                            )
        fig.add_trace(go.Scatter(x=w, y=20 * np.log10(abs(bessel.j0(2 * np.pi * w))), showlegend=False), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=(np.linspace(0, int(len(amplis) / 2) + 0.01, num=len(tot))) * resolution, y=tot,
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatterpolar(theta=np.linspace(0, 360, 360), r=afstand_per_hoek, mode='lines', showlegend=False),
            row=2, col=1)
        fig.add_trace(go.Heatmap(
            z=amplis,
            x=x,
            y=y,
            colorscale='Jet',
            zmin=mini,
            zmax=maxi
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=x_setup[-1 - S:-1],
            y=y_setup[-1 - S:-1],
            mode='markers',
            name='scatterers',
            marker_color='blue',
            marker=dict(size=3)
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[x_setup[-1]],
            y=[y_setup[-1]],
            name='antenna',
            mode='markers',
            text=str(ampl[int(len(ampl) / 2)][int(len(ampl[0]) / 2)]),
            marker_symbol="circle-x",
            marker_color="red",
            marker_line_color="white",
            marker_line_width=0.5,
            marker_size=5
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=x_setup[:M],
            y=y_setup[:M],
            name='BS',
            mode='markers',
            marker_color='red',
            marker=dict(size=3)
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[0, 800],
            y=[0, 800],
            mode='markers',
            name='dom',
            opacity=1,
            marker_color='white',
            marker=dict(size=3)
        ), row=1, col=2)
        fig.update_layout(autosize=False,
                          width=1550,
                          height=800,
                          title=fig_header,
                          font=dict(size=10),
                          polar=dict(radialaxis=dict(visible=False)),
                          polar2=dict(radialaxis=dict(visible=False)))
        plotly.offline.plot(fig, filename=file_name)
        fig.show()
    else:
        fig = go.Figure(data=go.Heatmap(
            z=amplis,
            x=x,
            y=y,
            colorscale='Jet',
            zmin=mini,
            zmax=maxi
        ))

        fig.add_trace(go.Scatter(
            x=x_setup[-S - 1:-1],
            y=y_setup[-S - 1:-1],
            mode='markers',
            name='scatterers',
            marker_color='blue',
            marker=dict(size=3)
        ))

        fig.add_trace(go.Scatter(
            x=[x_setup[-1]],
            y=[y_setup[-1]],
            name='antenna',
            mode='markers',
            marker_symbol="y-down",
            marker_line_color="red",
            marker_line_width=2,
            marker_size=5,
            text=str(ampl[int(len(ampl) / 2)][int(len(ampl[0]) / 2)])
        ))
        fig.add_trace(go.Scatter(
            x=x_setup[:M],
            y=y_setup[:M],
            name='BS',
            mode='markers',
            marker_color='red',
            marker=dict(size=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 800],
            y=[0, 800],
            mode='markers',
            name='dom',
            opacity=1,
            marker_color='white',
            marker=dict(size=3)
        ))
        fig.update_layout(autosize=False,
                          width=850,
                          height=800,
                          title=fig_header,
                          font=dict(size=10))
        plotly.offline.plot(fig, filename=file_name)
        fig.show()















































'''gemiddelde over 4 quadranten (werkt alleen met een HDX'''
aantal_runs = 4
UE_quad_x = [np.random.randint(int(len(x)/2), len(x)),
             np.random.randint(0, int(len(x)/2)),
             np.random.randint(0, int(len(x)/2)),
             np.random.randint(int(len(x)/2), len(x))]
UE_quad_y = [np.random.randint(int(len(y)/2), len(y)),
             np.random.randint(int(len(y)/2), len(y)),
             np.random.randint(0, int(len(y)/2)),
             np.random.randint(0, int(len(y)/2))]
gem_ampli = ampl/5
for r in range(aantal_runs):
    hq = np.array(HDX[UE_quad_y[r] * len(y) + UE_quad_x[r]])
    x_setup.append(x[UE_quad_x[r]])
    y_setup.append(y[UE_quad_y[r]])
    wHq = hq.conj().T
    ampl2 = np.zeros(shape=(len(x), len(y)))
    for a in range(len(y)):
        for b in range(len(x)):
            Hdx = np.array(HDX[a * len(y) + b])
            desired_signal = np.matmul(wHq, Hdx) / np.sqrt(M)
            #                  * signal_wave
            # sr = [cmath.polar(i) for i in desired_signal]
            # amplitudes = [item[0] if int(item[1]) == 0 else -item[0] for item in sr]
            # ratio_out_vs_rt_ampl = amplitudes[int(len(amplitudes) / 2) + 1] / signal_wave[int(len(signal_wave) / 2) + 1]
            # if ratio_out_vs_rt_ampl < 0:
            #     ampl[a, b] = -270
            # else:
            ampl2[a, b] = abs(desired_signal)
        # print(str(r+1)+'-'+str(a))
    title = './result plots/' + '(' + datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M") + ')-geo_model--M=' + str(M) + '_S=' + str(S) + '_dM=' + str(dM) + '_K=' + str(
        1) + '_resolution=' + str(resolution) + '_q' + str(r+1) + '.html'
    plotter(ampl2, title, header+'_q'+str(r+1), pl, resolution, S, M, x_setup, y_setup, x, y, wHq, all_in=True)
    gem_ampli = gem_ampli + ampl2/5
title = './result plots/' + '(' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + ')-geo_model_gem_quad--' + str(M) + '_S=' + str(S) + '_dM=' + str(dM) + '_K=' + str(
        1) + '_resolution=' + str(resolution) + '_gem.html'
#ziet dat ge hier de juiste setups meegeeft
plotter(gem_ampli, title, header+'_gem', pl, resolution, S, M, x_setup[:-4], y_setup[:-4], x, y, all_in=True)
#!!!!!!!!!!!!!!!plotter hierboven gedefinieerd!!!!!!!!!!!!!!!!!


