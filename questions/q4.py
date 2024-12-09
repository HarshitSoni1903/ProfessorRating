# -*- coding: utf-8 -*-
"""
@author: sonih
"""

from imports import pd, mannwhitneyu

def q4(num,tag):
    
    tag_mf = pd.concat([num[2],num[6],num[7],tag], axis=1, ignore_index=True)
    tag_mf.dropna(inplace=True)
    tag_mf = tag_mf.loc[tag_mf[1]!=tag_mf[2]]
    tag_mf = tag_mf.loc[tag_mf[0]>5]
    tag_col = [i for i in range(3,23)]
    tag_mf[tag_col] = tag_mf[tag_col].div(tag_mf[0], axis=0)
    male = tag_mf.loc[tag_mf[1]==1]
    fem = tag_mf.loc[tag_mf[1]==0]
    pval = {}
    for i in tag_col:
        _,p = mannwhitneyu(male[i], fem[i], alternative='two-sided')
        pval[p]=i-2
    return pval

if __name__ == '__main__':
    num = pd.read_csv('data/rmpCapstoneNum.csv', header=None)
    tags = pd.read_csv('data/rmpCapstoneTags.csv', header=None)
    print(q4(num,  tags))