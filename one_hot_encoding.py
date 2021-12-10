"""This module applies one hot encoding PLS on a dataset"""
import numpy as np
from math import e
from k_PLS import PLS1

def separate_data(X,y,i):
    """Separates data into training and testing set
    
    Input:  X       - an nxm matrix of a dataset
            y       - a column extracted from the dataset (the one of interest) 
            i       - the row of interest
    Output: Xtrain  - the X data with the row of interest taken outs
            X[i]    - The taken out row 
            Ytrain  - The array with the data point corresponding to i taken out
            Y[i]    - the taken out value"""
            
    if i == 0:
        return X[1:], y[1:], X[i], y[i]
    elif i == y.size-1:
        return X[:i],y[:i],X[i],y[i]
    
    Xtrain = np.concatenate((X[:i],X[i+1:]))
    Ytrain = np.concatenate((y[:i],y[i+1:]))
    
    return Xtrain,Ytrain, X[i], y[i]

def PLS_OHE(data,h):
    """Applies one hot encoding PLS on the dataset
    
    Input:  data - the dataset class (contains dataset, labels, number of 
                                      classes, and length of each class)
            h    - the number of iterations that should be taken
    Output: CM   - confusion matrix of the dataset
    """
    
    """Setting variables"""
    dataset = data.X0
    n,m = dataset.shape[0], dataset.shape[1]
    k = data.K
    
    """Setting labels"""
    labels = [] #change data labels into 0s and 1s
    for i in range(1,k+1):
        labels = labels + [i for x in range(data.K0[i-1])]
    labels = np.array(labels)
    
    Ktrain = np.zeros(k) #the number of observations in each training data class 
    CM = np.zeros((k,k)) #confusion matrix
    R = np.zeros((m,k)) #PLS model coefficients
    ypred = np.zeros(n) #predicted y values
    
    for i in range(n):
        Xtrain,Ytrain, xtest, ytest = separate_data(dataset,labels,i)
        
        """Changing the labels into 1, 2, 3"""
        for j in range(k):
            Ktrain[j] = len([x for x in Ytrain if x == j+1])
        Ktrain = Ktrain.astype(int)
        """Nb and Ne are to change the class values into 0 and 1"""
        Nb = 0          #beginning
        Ne = Ktrain[0]  #end
        
        """Determining the PLS model coefficients"""
        for j in range(k):
            y = np.zeros(sum(Ktrain))
            y[Nb:Ne] = 1
            if j < k-1:
                Nb += Ktrain[j]
                Ne += Ktrain[j+1]
            R[:,j] = PLS1(Xtrain,y,h)
        
        z = xtest @ R
        zsum = np.sum(e**z)
        zsoftmax = e**z / zsum
        y_hat = np.where(zsoftmax == max(zsoftmax))[0][0]
        ypred[i] = y_hat
        CM[ytest-1,y_hat] += 1
    return CM
