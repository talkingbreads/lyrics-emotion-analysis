# basic
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pickle

# preprocess
import re

# visualize
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy.interpolate import interpn
from sklearn.decomposition import PCA
from math import ceil


def tag_sent(cur_sent, future_sent, cur_sent_prob, future_sent_prob):

  valid_sents = ['positive', 'neutral', 'negative']
  if cur_sent not in valid_sents or future_sent not in valid_sents:
    return 'etc'

  # pos -> pos, neg -> neg, neutral -> neutral
  if cur_sent == future_sent:
    sent_tag = cur_sent

  # positive -> neatral (positive)
  # positive -> negative (precarious)
  elif cur_sent == 'positive':
    sent_tag = 'positive' if future_sent ==' neutral' else 'precarious'

  # neg -> neatral (improving)
  # neg -> positive (hopeful)
  elif cur_sent == 'negative':
    sent_tag = 'improving' if future_sent == 'neutral' else 'hopeful'

  # neutral -> positive (hopeful)
  # neutral -> negative (precarious)
  else:
    sent_tag = 'hopeful' if future_sent == 'positive' else 'precarious'

  return sent_tag


def compute_density_values(x, y, txt, bins):
    x, y, txt = np.asarray(x), np.asarray(y), np.asarray(txt)

    # compute density based on 2D histogram
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
        fill_value=0
    )
    z[np.isnan(z)] = 0.0

    # sort data based on density (to draw high-density plots later)
    idx = z.argsort()
    x_sorted = x[idx]
    y_sorted = y[idx]
    z_sorted = z[idx]
    txt_sorted = txt[idx]

    # scale density value to (R,G,B) scale
    z_max, z_min = z_sorted.max(), z_sorted.min()
    scaled_z = (z_sorted - z_min) / (z_max - z_min + 1e-8) * 255

    return x_sorted, y_sorted, z_sorted, txt_sorted, scaled_z


def compute_density_contour(x_sorted, y_sorted, n_contour):
    xy = np.vstack([x_sorted, y_sorted])
    kde = gaussian_kde(xy)

    x_min, x_max, y_min, y_max = min(x_sorted), max(x_sorted), min(y_sorted), max(y_sorted)

    x_grid = np.linspace(-1, 1, n_contour)
    y_grid = np.linspace(0,  1, n_contour)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([X.ravel(), Y.ravel()])

    # apply KDE results to grid
    Z = kde(grid_coords).reshape(X.shape)
    #Z[Z < 0.01] = np.nan  # ex. threshold = 0.01

    return x_grid, y_grid, Z


def draw_density_contour_plot(
    x, y, txt,
    noise_scale=0.01,
    n_contour=10,
    marker_size=12, marker_line_width=1,
    cm='Greys',
    title='', axis=['x', 'y'],
    w=800, h=800, 
):

    x, y = x.copy(), y.copy()

    x = np.array(x) + np.random.normal(-noise_scale, noise_scale, size=len(x))
    y = np.array(y) + np.random.normal(-noise_scale, noise_scale, size=len(y))

    x_grid, y_grid, Z = compute_density_contour(x, y, n_contour)

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=x_grid,
        y=y_grid,
        z=Z,
        colorscale=cm,
        opacity=1,
        contours=dict(showlabels=False),
        showscale=False,
        line=dict(width=0),
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        text=txt,
        mode='markers',
        marker=dict(
            size=marker_size,
            color='white',
            line=dict(color='rgba(90,90,90,1)', width=marker_line_width),
        ),
        hoverinfo='text'
    ))

    fig.update_layout(
        title=title,
        autosize=False,
        width=w,
        height=h,
        xaxis=dict(title=f'{axis[0]}', range=[-1,1]),
        yaxis=dict(title=f'{axis[1]}',range=[0,1]),

        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)' 
    )

    fig.show()


def draw_density_contour_subplots(
    fig, 
    row, col, n_rows, n_cols,
    x, y, txt,

    noise_scale=0.01,
    bins=[10,10], n_contour=25,
    marker_size=12, marker_line_width=1,
    cm='Greys',
    axis=['x', 'y'], title=''
):

    x, y = x.copy(), y.copy()
    x = np.array(x) + np.random.normal(-noise_scale, noise_scale, size=len(x))
    y = np.array(y) + np.random.normal(-noise_scale, noise_scale, size=len(y))

    x_sorted, y_sorted, z_sorted, txt_sorted, scaled_z = compute_density_values(x, y, txt, bins)
    x_grid, y_grid, Z = compute_density_contour(x_sorted, y_sorted, n_contour)

    # Contour trace
    fig.add_trace(go.Contour(
        x=x_grid,
        y=y_grid,
        z=Z,
        colorscale=cm,
        opacity=1,
        contours=dict(showlabels=False),
        showscale=False,
        line=dict(width=0),
        hoverinfo='skip'
    ), row=row, col=col)

    # Scatter trace
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=y_sorted,
        text=txt_sorted,
        mode='markers',
        marker=dict(
            size=marker_size,
            color='white',
            line=dict(color='rgba(90,90,90,1)', width=marker_line_width),
        ),
        hoverinfo='text',
        showlegend=False
    ), row=row, col=col)

    fig.update_xaxes(title_text=axis[0], range=[-1,1], row=row, col=col)
    fig.update_yaxes(title_text=axis[1], range=[0,1],  row=row, col=col)


def draw_density_scatter_plot(
    x, y, txt,
    noise_scale=0.01,
    bins=[10,10],
    marker_size=12,
    title='', axis=['x', 'y'],
    w=800, h=800,
):

    x, y = x.copy(), y.copy()

    x = np.array(x) + np.random.normal(-noise_scale, noise_scale, size=len(x))
    y = np.array(y) + np.random.normal(-noise_scale, noise_scale, size=len(y))

    x_sorted, y_sorted, z_sorted, txt_sorted, scaled_z = compute_density_values(x, y, txt, bins)

    c = [
        'rgba({},{},{},{})'.format(
            int(cm.viridis(int(val))[0] * 255),
            int(cm.viridis(int(val))[1] * 255),
            int(cm.viridis(int(val))[2] * 255),
            cm.viridis(int(val))[3]
        )
        for val in scaled_z
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=y_sorted,
        text=txt_sorted,
        mode='markers',
        marker=dict(size=marker_size,color=c,),
        hoverinfo='text'
    ))
    fig.update_layout(
        title=title,
        autosize=False,
        width=w, height=h,
        xaxis=dict(title=f'{axis[0]}', range=[-1.2, 1.2]),
        yaxis=dict(title=f'{axis[1]}', range=[-0.2, 1.2]),

        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.show()