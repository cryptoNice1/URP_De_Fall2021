# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:01:49 2021

@author: Melanie

This module performs a k-PLS model on a dataset 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def standard_deviation(array,mean):
    """Calculates sample standard deviation"""
    total = 0
    n = array.size
    for i in range(n):
        total+=(array[i] - mean)**2
    return (total/(n-1))**0.5

def matrix_standardize(X):
    """Standardizes a data set
    
    Input:  X - An nxm matrix representing m variables and n observations
    Output: X - A standardized version of X
    
    Standardizing a variable means centering it around 0 with standard 
    deviation = 1. For a variable m, standardized m = (m-mean(m))/stv(m)
    """
    new_matrix = np.ones((X.shape[0],X.shape[1])).T
    X = X.T
    m = X.shape[0]
    for i in range(1,m):
        x_mean = np.average(X[i])
        stv = standard_deviation(X[i],x_mean)
        X[i] = (X[i] - x_mean)
        new_matrix[i] = X[i]/stv
    return new_matrix.T

def PLS1(X,y,h):
    """ Computes the regression coefficients to predict y from X.
    
    Input:  X - nxm matrix of a full data set, n variables with m observations
            y - mx1 vector telling which sample a data point comes from
    Output: r - regression coefficients
    
    PLS is used when the matrix does not have full rank (ie. there are more 
    variables than data points [or] X cannot be inverted). 
    """
    m = X.shape[1]
    P = np.zeros((m,h))
    W = np.zeros((m,h))
    R = np.zeros((m,h))
    Q = np.zeros((h,1))
    
    for i in range(h):
        w = X.T @ y
        w = w/np.linalg.norm(w)
        r = w
        if i > 0:
            for j in range(i):
                r = r - (P[:,j].T @ w) * R[:,j]
        t = X @ r 
        q = y.T @ t / (t.T @ t)
        y = y - t * q
        
        P[:,i] = X.T @ t / (t.T @ t)
        R[:,i] = r
        Q[i] = q
        W[:,i] = w
        
    r = R @ Q
    return r.flatten()

def get_prediction(X_standardized,y_original,y_standardized,h,X):
    """Gets the model prediction
    
    Input:  X_standardized  - standardized data set 
            y_original      - oringinal data
            y_standardized  - standardized prediction data
    Output: y_0             - the predicted data
    
    Knowing X and y, use PLS1 to compute regression coefficients (r). We then use
    y_predicted = Xr. However, the data was standardized, so y_predicted is then
    reverse standardized. This y is then compared to the initial y for accuracy.
    """
    y_predict = X @ PLS1(X_standardized,y_standardized,h)
    y_0 = y_predict * np.std(y_original) + np.average(y_original)
    n = y_0.size
    
    for i in range(n):
        a = y_0[i]
        if a < 1.5:
            y_0[i] = 1
        elif a < 2.5:
            y_0[i] = 2
        elif a < 3.5:
            y_0[i] = 3
        elif a < 4.5:
            y_0[i] = 4
        else:
            y_0[i] = 5
    return y_0

def confusion_matrix(y_original,y_predicted):
    """Constructs a confusion matrix based on data
    
    Input:  y_original  - the original data column 
            y_predicted - the predicted values for that column 
    Output: CM          - the confusion matrix
    
    The rows represent the True classes (given labels) and the columns represent 
    the Predicted classes. 
    """
    m = y_original.size
    CM = np.zeros((m,m))
    y_o = y_original.astype(int)
    y_p = y_predicted.astype(int)
    for i in range(m):
        CM[y_o[i]-1, y_p[i]-1]+=1
    return CM

if __name__ == "__main__":
    f = "RamanData_Burn.txt"
    df = pd.read_csv(f, delimiter = "\t", header = None)
    
    """Setting up X and y"""
    y = np.array(df[0])*1.0
    df[0] = np.ones(y.size)
    X = np.array(df)
    
    X_s = matrix_standardize(X)
    y_s = (y - np.average(y)) / np.std(y)
    
    """Running program"""
    n = 40
    accuracies = np.zeros(40)
    for i in range(1,n+1):
        y_predict = get_prediction(X_s, y, y_s, i, X)
        CM = confusion_matrix(y,y_predict)
        ACC = np.sum(np.diagonal(CM)) / np.sum(CM)
        accuracies[i-1] = ACC
    
    """Plotting results"""
    x = np.linspace(1,n,n)
    plt.plot(x,accuracies)
