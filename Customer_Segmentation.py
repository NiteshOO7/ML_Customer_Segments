# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:01:35 2018

@author: nitesh.yadav
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import visuals as vs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def Data_Load():
    """loads data from CSV file"""
    try:
        full_data_frame = pd.read_csv(r"C:\Users\nitesh.yadav\Desktop\customer_segments\customer_data.csv")
    except FileNotFoundError:
        print("File 'customer_data.csv' does not exist, please check the provided path.")
    return full_data_frame

def Sample_Selection(data_frame):
    """To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail."""
    indices = [52, 205, 369]
    samples = pd.DataFrame(data_frame.loc[indices], columns = data_frame.keys()).reset_index(drop = True)
    return samples

def Feature_Visualization(data_frame):
    """To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data"""
    pd.plotting.scatter_matrix(data_frame, alpha = 0.3, figsize = (14, 8), diagonal = 'kde')
    
def Preprocess_Data(data_frame, samples):
    """Preprocess the data to create a better representation of customers by performing a scaling on the data"""
    log_data = np.log(data_frame)
    log_samples = np.log(samples)
    return log_data, log_samples

def Remove_Outliers(normData):
    """Detecting outliers in the data is extremely important in the data preprocessing step of any analysis"""
    for feature in normData.keys():
        Q1 = np.percentile(normData[feature], 25)
        Q3 = np.percentile(normData[feature], 75)
        step = 1.5 * (Q3 - Q1)
        print("Data points considered outliers for the feature '{}':".format(feature))
        display(normData[~((normData[feature] >= Q1 -step) & (normData[feature] <= Q3 + step))])
    outliers = [38, 57, 65, 66, 75, 81, 86, 95, 96, 98, 109, 128, 137, 142, 145, 154, 161, 171, 175, 183, 184, 187, 193, 203, 218, 233, 264, 285, 289, 304, 305, 325, 338, 343, 353, 355, 356, 357, 412, 420, 429, 439]
    good_data = normData.drop(normData.index[outliers]).reset_index(drop = True)
    return good_data

def Feature_Transform(good_data, log_samples):
    """Principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data"""
    pca = PCA(n_components = 6)
    pca.fit(good_data)
    pca_samples = pca.transform(log_samples)
    pca_results = vs.PCA_Results(good_data, pca)
    display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))
    pca = PCA(n_components = 2)
    pca.fit(good_data)
    reduced_data = pca.transform(good_data)
    pca_samples = pca.transform(log_samples)
    reduced_data = pd.DataFrame(reduced_data, columns = ['Dimention 1', 'Dimention 2'])
    display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))
    vs.Biplot(good_data, reduced_data, pca)
    return  reduced_data, pca_samples, pca

def Clustering(reduced_data, pca_samples):
    """Fit a clustering algorithm to the reduced_data and assign it to clusterer"""
    clusterer = KMeans(n_clusters = 5, random_state = 0)
    clusterer.fit(reduced_data)
    preds = clusterer.predict(reduced_data)
    centers = clusterer.cluster_centers_
    pca_samples = clusterer.predict(pca_samples)
    score = silhouette_score(reduced_data, preds)
    print("Cluster Centers:", centers)
    print("Score:", score)
    # Display the results of the clustering from implementation
    #vs.Cluster_Results(reduced_data, preds, centers, pca_samples)
    return centers

def Data_Recovery(pca, centers, normData):
    """ Apply the inverse transform to centers using pca.inverse_transform and assign the new centers to log_centers"""
    log_centers = pca.inverse_transform(centers)
    true_centers = np.exp(log_centers)
    # Display the true centers
    segments = ['Segment {}'.format(i) for i in range(0, len(centers))]
    true_centers = pd.DataFrame(np.round(true_centers), columns = normData.keys())
    true_centers.index = segments
    display(true_centers)
    
    
    
    

    
    
        