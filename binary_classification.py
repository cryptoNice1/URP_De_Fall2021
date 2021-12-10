# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 19:00:42 2021

@author: Melanie

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from k_PLS import matrix_standardize

class Binary_classes:
    """A data structure for the classes"""
    
    def __init__(self,f,c1, c2):
        """Initializes the 2 classes from the given dataset
        
        Input:      df  - a pandas dataframe representation of the data
                    c1  - the first class number
                    c2  - the second class number
        Variables:  K0  - list of number of observations in c1 and c2
                    X0  - the new matrix of just the two classes
                    lab - array of just the data labels"""
        
        self.K = 2
        """Reading in data"""
        df = pd.read_csv(f, delimiter = "\t", header = None)
        self.lab = np.array(df[0])
        
        """Creating new dataset with just the two classes"""
        class1 = df[df[0] == c1].to_numpy()
        class2 = df[df[0] == c2].to_numpy()
        self.K0 = [class1.shape[0],class2.shape[0]]
        self.X0 = np.concatenate((class1, class2),axis=0)
        
        self.X0[:,0] = np.ones(sum(self.K0))
        self.X0 = matrix_standardize(self.X0)
        
    def __str__(self):
        print(pd.DataFrame(self.X0))
        print("K0:",self.K0)
        X = self.X0
        print("X0:",X.shape[0],"x",X.shape[1])
        print("K")
        return""
