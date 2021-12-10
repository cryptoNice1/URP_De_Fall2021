# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 20:45:14 2021

@author: Melanie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binary_classification import One_versus_one, One_versus_all
from One_Hot_Encoding import PLS_OHE

def find_best_model(data,n):
    """Runs a OHE k-PLS model on the dataset with a various amount of iterations
    to determine the value of h that has the highest accuracy."""
    accuracies = np.zeros(n)
    for i in range(1,n+1):
        CM = PLS_OHE(data,i,False)
        accuracies[i-1] = np.sum(np.diagonal(CM)) / np.sum(CM)
    return np.argmax(accuracies)

def find_area(TPR, FPR):
    n = TPR.shape[0]
    area = 0
    for f1, f2, x1,x2 in zip(TPR[:n],TPR[1:],FPR[:n],FPR[1:]):
        area += (f2 + f1)*(x1-x2)/2
    return area


def ROC_curve(TPR,FPR,index=0):
    plt.subplot(121)
    area = find_area(TPR[:,index],FPR[:,index])
    text_string = "AUC: {:.2f}".format(area)
    plt.text(0.7,0, text_string)
    
    plt.plot(FPR[:,index],TPR[:,index])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")

def show_metrics(TPR,FPR,ACC, index=0):
    plt.subplot(122)
    plt.plot(threshold_values,1-FPR[:,index],label="spe")
    plt.plot(threshold_values,TPR[:,index],label = "sen")
    plt.plot(threshold_values,ACC[:,index],label = "acc")
    plt.xlabel("threshold value")
    plt.ylabel("Percentage")
    plt.legend(loc = "best",fontsize = 8)
    plt.title("Metrics")

def multi_metrics(TPR,FPR,i):
    acc = np.amax(ACC[:,i])
    index = np.argmax(ACC[:,i])
    spe = 1 - FPR[index,i]
    sen = TPR[index,i]
    labels = ["max acc {:.2f}".format(acc),"spe {:.2f}".format(spe),"sen {:.2f}".format(sen)]
    
    plt.subplot(1,2,2)
    plt.plot(threshold_values,ACC[:,0],label = labels[0])
    plt.plot(threshold_values,1-FPR[:,0],label=labels[1])
    plt.plot(threshold_values,TPR[:,0],label = labels[2])
    plt.xlabel("Threshold value")
    plt.ylabel("Percentage")
    plt.legend(loc = "best")
    plt.title("Metrics")
    plt.tight_layout()

"""Getting input and setting constants"""
df = pd.read_csv("RamanData_Burn.txt", delimiter = "\t", header = None)
upper_bound_h = 10
threshold_values = np.linspace(0,1,1000)

"""=============================================================================
# ONE VERSUS ONE CLASSIFICATION
============================================================================="""

show_model = False
if not show_model:
    print("Binary classification results:")
    print("classes |  acc  |  sen  |  spe  ")

for c1 in range(1,6):
    for c2 in range(c1+1,6):
        big_title = "Classes "+str(c1)+" and "+str(c2)
        figure_name = big_title+".png"
        
        data = One_versus_one(df, c1, c2)
        n_best = find_best_model(data,upper_bound_h)
        TPR, FPR, ACC = PLS_OHE(data,n_best+1,True)
        
        if show_model:
            plt.figure(figsize=(8,3))
            plt.suptitle(big_title,fontsize = 15)
            ROC_curve(TPR,FPR)
            show_metrics(threshold_values,TPR,FPR,ACC)
            
            plt.subplots_adjust(top=0.85)
            plt.savefig(figure_name,dpi=300,edgecolor='none',bbox_inches = "tight")
        else:
            acc = np.amax(ACC[:,0])
            index = np.argmax(ACC[:,0])
            spe = 1 - FPR[index,0]
            sen = TPR[index,0]
            print(" ",c1,"", c2,"    {:.2f}    {:.2f}    {:.2f}".format(acc,sen,spe))


"""=============================================================================
# ONE VERSUS ALL CLASSIFICATION
============================================================================="""
data = One_versus_all(df)
n_best = find_best_model(data,upper_bound_h)
TPR, FPR, ACC = PLS_OHE(data,n_best+1,True)
area = find_area(TPR,FPR)
k = data.K

for i in range(data.K):
    plt.figure(figsize=(8,3))
    ROC_curve(TPR,FPR,i)
    multi_metrics(TPR,FPR,i)
    big_title = "{} vs all".format(i+1)
    plt.suptitle(big_title,fontsize = 15)
    plt.subplots_adjust(top=0.85)
    plt.savefig(big_title+".png",dpi=300,edgecolor='none',bbox_inches = "tight")
    plt.close()
