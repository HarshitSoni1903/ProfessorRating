# -*- coding: utf-8 -*-
"""
@author: sonih
"""

from imports import np,pd,plt
from imports import mannwhitneyu
from imports import ecdf

def ec(num,qual):
    
    newtag = pd.concat([num[0], qual[0]], axis=1, ignore_index=True)
    newtag.dropna(inplace=True)
    field = set(newtag.loc[newtag[1].str.contains('comput', case=False),1])
    field.update(set(newtag.loc[newtag[1].str.contains('math', case=False),1]))
    field.update(set(newtag.loc[newtag[1].str.contains('bio', case=False),1]))
    field.update(set(newtag.loc[newtag[1].str.contains('chem', case=False),1]))
    field.update(set(newtag.loc[newtag[1].str.contains('physics', case=False),1]))
    field.update(set(newtag.loc[(newtag[1].str.lower() =='science'),1]))
    field.update(set(newtag.loc[newtag[1].str.contains('stati', case=False),1]))
    field.update(set(newtag.loc[newtag[1].str.contains('astro', case=False),1]))
    field.update(set(newtag.loc[(~newtag[1].str.contains('english', case=False))&(newtag[1].str.contains('eng', case=False))]))
    field.update(set(newtag.loc[newtag[1].str.contains('info', case=False),1]))
    field.update(set(newtag.loc[newtag[1].str.contains('info', case=False),1]))
    field.update(set(newtag.loc[newtag[1].str.contains('tech', case=False),1]))
    field.update(set(newtag.loc[newtag[1].str.contains('information', case=False),1]))
    
    stem = newtag.loc[newtag[1].isin(field),0]
    nonstem = newtag.loc[~newtag[1].isin(field),0]
    x_stem, y_stem = ecdf(stem)
    x_nonstem, y_nonstem = ecdf(nonstem)
    fig,ax = plt.subplots(1,2)
    fig.suptitle("Extra Credit: Empirical CDF-Stem vs NonStem ")
    ax[0].plot(x_stem, y_stem, label="Stem", lw=2)
    ax[0].plot(x_nonstem, y_nonstem, label="Nonstem", lw=2)
    ax[0].set_xlabel("Values")
    ax[0].set_ylabel("ECDF")
    ax[0].set_title("Before random undersampling")
    ax[0].legend()
    ax[0].grid(alpha=0.5)
    
    
    rng = np.random.RandomState(17865635)
    nonstem = nonstem.loc[rng.random(len(nonstem))<0.3]
    stat, p = mannwhitneyu(stem, nonstem)
    
    x_stem, y_stem = ecdf(stem)
    x_nonstem, y_nonstem = ecdf(nonstem)
    ax[1].plot(x_stem, y_stem, label="Stem", lw=2)
    ax[1].plot(x_nonstem, y_nonstem, label="Nonstem", lw=2)
    ax[1].set_xlabel("Values")
    ax[1].set_ylabel("ECDF")
    ax[1].set_title("After random undersampling")
    ax[1].legend()
    ax[1].grid(alpha=0.5)    
    
    plt.show()

    return stat,p

if __name__ == '__main__':
    num = pd.read_csv('./data/rmpCapstoneNum.csv', header=None)
    qual = pd.read_csv('./data/rmpCapstoneQual.csv', header=None)
    print(ec(num,qual))