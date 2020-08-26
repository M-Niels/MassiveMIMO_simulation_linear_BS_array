import cmath
from numpy.random import rand
import numpy as np
import scipy.special as bessel
import plotly.io as pio
import angular_distribution as ad
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def plotter(ampl, file_name, fig_header, resolution, S, M, x_setup, y_setup, UE_x, UE_y, x, y, wH=None, all_in=True, mini=None,
            maxi=None):
    """plots the calculated grid and the surrounding area"""
    UE = ampl[UE_y][UE_x]
    amplis = [20 * np.log10(i / UE) for i in ampl]
    if mini is not None and maxi is None:
        maxi = max([max(i) for i in ampl])
    elif maxi is not None and mini is None:
        mini = min([min(i) for i in ampl])

    '''bubble'''
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
        while prev_val >= inter:
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

    if all_in:
        '''Avg intersection'''
        tot = amplis[int(len(x) / 2)][int(len(y) / 2):-1]
        for i in range(1, 360):
            angle = i * np.pi / 180
            x_step = np.cos(angle)
            y_step = np.sin(angle)
            ix = int(len(amplis) / 2)
            ij = int(len(amplis) / 2)
            graph = [amplis[ij][ix]]
            for j in range(int(len(amplis) / 2)-1):
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

        '''radi pat'''
        theta_deg = np.linspace(-180, 180 + 0.01, num=10000)
        theta_rad = np.linspace(-np.pi, np.pi + 0.01, num=10000)
        temp = [cmath.polar(i) for i in wH]  # wH = antenna weights
        amplitude = [i[0] for i in temp]
        phase = [i[1] for i in temp]
        rad_pat = []
        for i in range(len(wH)):
            rad_pat.append(amplitude[i] * np.exp(1j * (phase[i])) * np.exp(1j * i * np.pi * np.cos(theta_rad - np.pi / 2)))
        radi_pattern = sum(rad_pat)

        '''angular pwr'''
        ang_pwr = ad.weighted_power(S, M, x_setup, y_setup, 16, plotter='plotly')

        fig = make_subplots(rows=2, cols=4, column_widths=[250, 250, 250, 850],
                            specs=[[{"colspan": 3}, None, None, {"rowspan": 2}],
                                   [{"type": "polar"}, {"type": "Barpolar"}, {"type": "polar"}, None]])
        w = np.linspace(0, int(len(amplis) / 2) + 0.01, num=1000)
        w = w * resolution
        fig.add_trace(go.Scatter(x=w, y=20 * np.log10(abs(bessel.j0(2 * np.pi * w))), showlegend=False), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=(np.linspace(0, int(len(amplis) / 2) + 0.01, num=len(tot))) * resolution, y=tot,
                                 showlegend=False), row=1, col=1)
        fig.add_trace(ang_pwr, row=2, col=2)
        fig.add_trace(go.Scatterpolar(theta=theta_deg, r=abs(radi_pattern), mode='lines', showlegend=False), row=2, col=1)
        fig.add_trace(
            go.Scatterpolar(theta=np.linspace(0, 360, 360), r=dist_by_angle, mode='lines', showlegend=False),
            row=2, col=3)

        fig.add_trace(go.Heatmap(
            z=amplis,
            x=x,
            y=y,
            colorscale='Jet',
            zmin=mini,
            zmax=maxi
        ), row=1, col=4)
        fig.add_trace(go.Scatter(
            x=x_setup[-1 - S:-1],
            y=y_setup[-1 - S:-1],
            mode='markers',
            name='scatterers',
            marker_color='blue',
            marker=dict(size=3)
        ), row=1, col=4)
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
        ), row=1, col=4)
        fig.add_trace(go.Scatter(
            x=x_setup[:M],
            y=y_setup[:M],
            name='BS',
            mode='markers',
            marker_color='red',
            marker=dict(size=3)
        ), row=1, col=4)
        fig.add_trace(go.Scatter(
            x=[0, 800],
            y=[0, 800],
            mode='markers',
            name='dom',
            opacity=1,
            marker_color='white',
            marker=dict(size=3)
        ), row=1, col=4)
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
    return [np.var(dist_by_angle), np.average(dist_by_angle), np.var(values_edge_bubble), np.average(values_edge_bubble)]





def IQ(array):
    I = []
    Q = []
    for i in range(len(array)):
        if i % 2 == 0:
            I.append(array[i])
        else:
            Q.append(array[i])
    if len(I) != len(Q):
        I = Q = None
    return I, Q


# def data_to_voltage_lvl(array):
#     for i in range(len(array)):
#         if array[i] == 0:
#             array[i] = -1  # /sqrt(2)
#         # elif array[i] == 1:
#         #     array[i] = 1/sqrt(2)


def data_to_voltage_lvl(array, tech):
    technr = np.log2(np.sqrt(tech))
    if len(array) % technr != 0:
        return [1]
    array_a = []
    temp = technr
    n = 0
    for i in range(len(array)):
        if i % technr == 0:
            array_a.append(1 - 2 ** technr + n * 2)
            temp = technr
            n = 0
        n += array[i] * 2 ** (temp - 1)
        temp -= 1
    array_a.append(1 - 2 ** technr + n * 2)
    return [int(i) for i in array_a[1:]]


def avg(array):
    gem = 0
    for i in range(0, len(array)):
        gem += array[i]
    return gem / len(array)


def rho(x):
    return bessel.j0(2 * np.pi * x)  # Bessel function 1st kind, 0th order


def hd(dx, h):
    e = 0.2 * rand(1)[0] + 0.9  # innovation factor
    return rho(dx) * h + (np.sqrt(1 - (abs(rho(dx))) ** 2) * e)


'''Test zone'''
# a = [0,1,1,1,0,1,1,0,1,0,0,0,1,0,0,1]
# a = data_to_voltage_lvl(a, 16)
# print(a)