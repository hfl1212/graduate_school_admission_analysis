"""
Implements functions to create boxplots and heatmaps for research question 2.
"""
import plotly.graph_objects as go
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


def boxplots_testscores_vs_admission(df):
    '''
    Plots three box plots for three scores (GRE, TOEFL, CGPA) versus the
    chance of admission.
    '''
    fig = make_subplots(subplot_titles=("Chance of Admit vs GRE Score",
                                        "Chance of Admit vs TOEFL Score",
                                        "Chance of Admit vs CGPA"),
                        rows=3, cols=1)
    gre_plot = go.Box(x=df['GRE Score'], y=df['Chance of Admit'])
    toefl_plot = go.Box(x=df['TOEFL Score'], y=df['Chance of Admit'])
    cgpa_plot = go.Box(x=df['CGPA'], y=df['Chance of Admit'])
    fig.append_trace(gre_plot, row=1, col=1)
    fig.append_trace(toefl_plot, row=2, col=1)
    fig.append_trace(cgpa_plot, row=3, col=1)
    fig.update_layout(height=1200, width=800,
                      title_text='Chance of Admit vs Scores')
    fig.write_image('chance_vs_scores.png')


def find_correlation(df):
    '''
    Creates the heatmap that shows the correlation between the variables.
    '''
    corr = df.corr()
    drop_upper = np.zeros_like(corr)
    drop_upper[np.triu_indices_from(drop_upper)] = True
    colormap = sb.diverging_palette(240, 10, n=8, as_cmap=True)
    plt.subplots(figsize=(8, 8))
    sb.heatmap(corr, cmap=colormap, linewidths=.1, annot=True, fmt='.2f',
               mask=drop_upper)
    plt.title('Correlation between Variables')
    plt.savefig('parameters_correlation.png')
