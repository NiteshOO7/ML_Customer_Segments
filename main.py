# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:00:50 2018

@author: nitesh.yadav
"""
import Customer_Segmentation as cs

def main():
    # load data from csv file
    full_data_frame = cs.Data_Load()
    print("Customer dataset has {} data points with {} variables each.".format(*full_data_frame.shape))
    full_data_frame.drop(['Region', 'Channel'], axis = 1, inplace = True)
    display(full_data_frame.describe())
    print("Customer dataset has {} data points with {} variables each.".format(*full_data_frame.shape))
    samples = cs.Sample_Selection(full_data_frame)
    # Visualize features
    cs.Feature_Visualization(full_data_frame)
    normData, log_samples = cs.Preprocess_Data(full_data_frame, samples)
    #display(normData)
    good_data = cs.Remove_Outliers(normData)
    #PCA
    reduced_data, pca_samples, pca = cs.Feature_Transform(good_data, log_samples)
    #Clustering
    centers = cs.Clustering(reduced_data, pca_samples)
    # Data Recovery
    cs.Data_Recovery(pca, centers, normData)

    
    
if __name__ == "__main__":
    main()

