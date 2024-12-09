# -*- coding: utf-8 -*-
"""
@author: sonih
"""
from imports import np,pd, ks_2samp

def q5(num):
    num = num.loc[num[6]!=num[7]]
    for i in [0,3,4,5,7]:
        num.drop(i, axis=1, inplace=True)
    num.dropna(inplace=True)
    num = num.loc[num[2]>5]
    male = num.loc[num[6]==1,1]
    fem = num.loc[num[6]==0,1]
    stat, p = ks_2samp(male, fem, alternative = 'two-sided')
    #Check if p value is less than alpha to establish significance
    if(p<0.005):
        print("The null hypothesis is rejected. Observed data is unlikely given chance.")
    else:
        print("The null hypothesis is not rejected.")
    n1 = male.shape[0]
    n2 = fem.shape[0]
    std1 = np.std(male)
    std2 = np.std(fem)
    std_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    mean_diff = np.mean(male) - np.mean(fem)
    d = mean_diff/std_pooled
    return p,d


if __name__ == '__main__':
    num = pd.read_csv('data/rmpCapstoneNum.csv', header=None)
    print(q5(num))