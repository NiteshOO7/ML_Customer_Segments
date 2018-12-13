# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:01:53 2018

@author: nitesh.yadav
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import Customer_Segmentation as cs

def PCA_Results(good_data, PCA):
    """Create a DataFrame of the PCA results, Includes dimension feature weights and explained variance, Visualizes the PCA results"""
    dimensions = ['Dimension {}'.format(i) for i in range(1, len(PCA.components_) + 1)]
    components = pd.DataFrame(np.round(PCA.components_, 4), columns = list(good_data.keys()))
    components.index = dimensions
    ratios = PCA.explained_variance_ratio_.reshape(len(PCA.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    fig, ax = plt.subplots(figsize = (14, 8))
    components.plot(ax = ax, kind = 'bar')
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation = 0)
    for i, ev in enumerate(PCA.explained_variance_ratio_):
        ax.text(i - 0.4, ax.get_ylim()[1] + 0.05, "Explained Variance\n    %.4f"%(ev))
    return pd.concat([variance_ratios, components], axis = 1)

def Cluster_Results(reduced_data, preds, centers, PCA_samples):
    """Visualizes the PCA-reduced cluster data in two dimensions, Adds cues for cluster centers and student-selected sample data"""
    predictions = pd.DataFrame(preds, columns = ['Cluster'])
    plot_data = pd.concat([predictions, reduced_data], axis = 1)
    fig, ax = plt.subplots(figsize = (14, 8))
    cmap = cm.get_cmap('gist_rainbow')
    for i, cluster in plot_data.groupby('Cluster'):
        cluster.plot(ax = ax, kind = 'scatter', x='Dimension 1', y = 'Dimension 2', color = cmap((i) * 1.0/(len(centers) - 1)), label = 'Cluster %i'%(i), s = 30);
    for i, c in enumerate(centers):
        ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', alpha = 1, linewidth = 2, marker = 'o', s = 200);
        ax.scatter(x = c[0], y = c[1], marker = '$%d$'%(i), alpha = 1, s = 100);
    ax.scatter(x = PCA_samples[:, 0], y = PCA_samples[:, 1], s = 150, linewidth = 4, color = 'black', marker = 'x');
    ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");

def Biplot(good_data, reduced_data, PCA):
    """Produce a biplot that shows a scatterplot of the reduced data and the projections of the original features.
    good_data: original data, before transformation. Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute
    return: a matplotlib AxesSubplot object (for any additional customization)
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot"""
    fig, ax = plt.subplots(figsize = (14, 8))
    ax.scatter(x = reduced_data.loc[:, 'Dimention 1'], y = reduced_data.loc[:, 'Dimention 2'], facecolors = 'b', edgecolors = 'b', s = 70, alpha = 0.5)
    feature_vectors = PCA.components_.T
    arrow_size, text_pos = 7.0, 8.0
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size * v[0], arrow_size * v[1], head_width = 0.2, head_length = 0.2, linewidth = 2, color = 'red')
        ax.text(v[0] * text_pos, v[1] * text_pos, good_data.columns[i], color = 'black', ha = 'center', va = 'center', fontsize = 18)
    ax.set_xlabel('Dimension 1', fontsize = 14)
    ax.set_ylabel('Demension 2', fontsize = 14)
    ax.set_title("PC plane with original feature projections.", fontsize = 16)
    return ax

def Chennel_Results(reduced_data, outliers, PCA_samples):
    """Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
    Data is labeled by "Channel" and cues added for student-selected sample data"""
    full_data_frame = cs.Data_Load()
    channel = pd.DataFrame(full_data_frame['channel'], columns = ['channel'])
    channel = channel.drop(channel.index[outliers]).reset_index(drop = True)
    labeled = pd.concat([reduced_data, channel], axis = 1)
    fig, ax = plt.subplots(figsize = (14, 8))
    cmap = cm.get_cmap('gist_rainbow')
    labels = ['Hotel/Restaurant/Cafe', 'Retailer']
    grouped = labeled.groupby('Channel')
    for i, channel in grouped:
        channel.plot(ax = ax, kind = 'scatter', x = 'Dimention 1', y = 'Dimention 2', color = cmap((i - 1) * 1.0/2), label = labels[i - 1], s = 30)
    for i, sample in enumerate(PCA_samples):
        ax.scatter(x = sample[0], y = sample[1], s = 200, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none')
        ax.scatter(x = sample[0] + 0.25, y = sample[1] + 3.0, marker = '$%d'%(i), alpha = 1, s = 125)
    ax.set_title("PCA-Reduced Data Labeled by 'Channel'\nTransformed Sample Data Circled")
        