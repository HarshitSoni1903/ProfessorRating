# -*- coding: utf-8 -*-
"""
@author: sonih
"""

from imports import np,pd, plt
from imports import LinearRegression, train_test_split, StandardScaler
from imports import mean_squared_error, r2_score


def q8(num,tags):
    tag_mf = pd.concat([num[0],num[2],num[6],num[7],tags], axis=1, ignore_index=True)
    tag_mf.dropna(inplace=True)
    tag_mf = tag_mf.loc[tag_mf[2]!=tag_mf[3]]
    tag_mf = tag_mf.loc[tag_mf[1]>5]
    
    tag_col = [i for i in range(4,24)]
    tag_mf[tag_col] = tag_mf[tag_col].div(tag_mf[1], axis=0)
    
    corr = tag_mf[tag_col].corr()
    corr.style.background_gradient(cmap='coolwarm')
    X_train, X_test, y_train, y_test = train_test_split(tag_mf[tag_col], tag_mf[0], test_size=0.2, random_state=17865635)
    
    sc = StandardScaler()
    X_sc_tr = sc.fit_transform(X_train)
    X_sc_te = sc.transform(X_test)
    
    leanR = LinearRegression()
    leanR.fit(X_sc_tr, y_train)
    pred = leanR.predict(X_sc_te)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    wght = leanR.coef_
    plt.figure()
    plt.title("Question 8: Linear Regrssion Model Weights")
    plt.bar([i+1 for i in range(len(wght))],wght, color = 'orange')
    plt.xticks([i+1 for i in range(len(wght))])
    plt.show()
    max_abs_index = np.argmax(np.abs(wght))
    return rmse,r2,max_abs_index,wght[max_abs_index]
            
if __name__ == '__main__':
    num = pd.read_csv('data/rmpCapstoneNum.csv', header=None)
    tags = pd.read_csv('data/rmpCapstoneTags.csv', header=None)
    print(q8(num,  tags))