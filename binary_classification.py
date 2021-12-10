# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 19:00:42 2021

@author: Melanie

"""
import pandas as pd
import numpy as np
from k_PLS import matrix_standardize

class One_versus_one:
    """One versus one data structure"""
    
    def __init__(self,df,c1, c2):
        """Initializes the 2 classes from the given dataset
        
        Input:      df  - the dataframe
                    c1  - the first class number
                    c2  - the second class number"""
        
        self.K = 2
        
        """Reading in data"""
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
        print("K:",self.K)
        return""
        
class One_versus_all:
    """A general data structure for the classes"""
    
    def __init__(self,df):
        """Initializes the 2 classes from the given class and the others
        
        Input:      f   - a pandas dataframe representation of the data"""
    
        self.K = 5
        
        """Reading in data"""
        self.lab = np.array(df[0])
        
        """Creating new dataset with just two classes"""
        self.K0 = []
        for i in range(1,6):
            self.K0.append(df[df[0] == i].to_numpy().shape[0])
            
        self.K = 5
        self.X0 = df.to_numpy()
        self.X0[:,0] = np.ones(sum(self.K0))
        self.X0 = matrix_standardize(self.X0)
        
    def __str__(self):
        print(pd.DataFrame(self.X0))
        print("K0:",self.K0)
        X = self.X0
        print("X0:",X.shape[0],"x",X.shape[1])
        print("K:",self.K)
        return""

if __name__ == "__main__":
    f = "RamanData_Burn.txt"
    print(One_versus_one(f,1,2))
    print(One_versus_all(f))
