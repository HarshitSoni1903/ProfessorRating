# -*- coding: utf-8 -*-
"""
@author: sonih
"""

from imports import pd, numname, tagname

#Import solution modules
from questions import q1,q4,q5,q7,q8,q9,q10,ec

#Supress warning for a cleaner output.
import warnings

warnings.filterwarnings("ignore")

#Load datasets
num = pd.read_csv('data/rmpCapstoneNum.csv', header=None)
tags = pd.read_csv('data/rmpCapstoneTags.csv', header=None)
qual = pd.read_csv('data/rmpCapstoneQual.csv', header=None)



def main():
    #Module 1 solves question 1,2,3
    print("Question 1")
    s1,p1,s2,p2,d = q1.q1(num)
    print("P-value: " +str(p1))
    print("Question 2")
    if(p2<0.005):
        print("The null hypothesis is rejected. Observed data is unlikely given chance.")
    else:
        print("The null hypothesis is not rejected.")
    print("P-value: " +str(p2))
    print("Question 3")
    print("Cohen's-d: " +str(d))
    
    #Module 4 solves question4
    print("Question 4")
    #Returns a dictionary object with p-values as keys and column number as the value.
    pval = q4.q4(num, tags)
    #To find the to 3 and bottom 3 columns, we will sort the p-values in increasing order and pict first 3 and last 3
    pvals = sorted(pval.keys())
    print("Top 3 significant features") 
    for i in pvals[:3]:
        print("Column: "+ str(tagname[pval[i]])+ ": P-value:  "+str(i))
    print("Least 3 significant features")
    for i in pvals[-3:]:
        print("Column: "+ str(tagname[pval[i]])+ ": P-value:  "+str(i))
    
    #Module q5 solves question 5 and 6
    print("Question 5")
    p5,d5 = q5.q5(num)
    print("P-value: " +str(p5))
    print("Question 6")
    print("Cohen's-d: " +str(d5))
    
    #Module q7 solves the question 7 and returns rmse, r2 for the linear regression model we built
    print("Question 7")
    rmse7,r27, ind,wei = q7.q7(num)
    print("RMSE" + str(rmse7))
    print("R2: "+ str(r27))
    print("Feature with largest absolute weight: " + numname[ind+1])
    print(wei)
    
    #Module q8 solves question 8 and returns rmse, r2 and the most contributing factor with wieght for the linear regression model
    print("Question 8")
    rmse8,r28,ind,wei = q8.q8(num,tags)
    print("RMSE: " + str(rmse8))
    print("R2: "+ str(r28))
    print("Feature with largest absolute weight: " + tagname[ind+1])
    print(wei)
    #Module q9 solves question 9 and returns rmse, r2 and the most contributing factor with wieght for the linear regression model
    print("Question 9")
    rmse9,r29,ind,wei = q9.q9(num,tags)
    print("RMSE" + str(rmse9))
    print("R2: "+ str(r29))
    print("Feature with largest absolute weight: " + tagname[ind+1])
    print(wei)
    #Module q10 solves question 8 and returns AU(RO)C score, classification report for the logistic regression model
    print("Question 10")
    roc_score, class_rep = q10.q10(num, tags)
    print("AU(RO)C:" + str(roc_score))
    print("Classification report")
    print(class_rep)
    
    #Extra credit
    print("Extra credit")
    statec,pec = ec.ec(num,qual)
    print(pec)
    
if __name__ == '__main__':
    main()