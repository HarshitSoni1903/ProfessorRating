# -*- coding: utf-8 -*-
"""
@author: sonih
"""

from imports import np,pd,plt
from imports import ks_2samp, mannwhitneyu
from imports import ecdf
def q1(num):
    
    #Filter out data which has both male and females as 0-0 or 1-1
    num = num.loc[num[6]!=num[7]]
    #Remove columns not needed for the test
    num.drop([1,3,4,5,7], axis=1, inplace=True)
    #Drop null values
    num.dropna(inplace=True)
    #Use data with number of rating more than 5
    num = num.loc[num[2]>5]
    #Select males and females
    male = num.loc[num[6]==1,0]
    fem = num.loc[num[6]==0,0]
    
    #Perform mann whitney u test with alternate hypothesis being pro-male bias in rating
    stat1, p1 = mannwhitneyu(male, fem, alternative = 'greater')
    
    #Check if p value is less than alpha to establish significance
    if(p1<0.005):
        print("The null hypothesis is rejected. Observed data is unlikely given chance.")
    else:
        print("The null hypothesis is not rejected.")
    
    plt.figure()
    plt.boxplot([male, fem], labels=["Males", "Females"])
    plt.ylabel("Values")
    plt.title("Question 1: Boxplot- Males vs Females")
    plt.show()
    #Calculate cohen's d to measure effect
    n1 = male.shape[0]
    n2 = fem.shape[0]
    std1 = np.std(male)
    std2 = np.std(fem)
    std_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    mean_diff = np.mean(male) - np.mean(fem)
    d = mean_diff/std_pooled
    stat_2,p_2 = ks_2samp(male, fem, alternative='two-sided')
    
    x_males, y_males = ecdf(male)
    x_fem, y_fem = ecdf(fem)
    plt.figure()
    plt.plot(x_males, y_males, label="Males", lw=2)
    plt.plot(x_fem, y_fem, label="Females", lw=2)
    plt.xlabel("Values")
    plt.ylabel("ECDF")
    plt.title("Question 2: Empirical CDF - Males vs Females")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    
    plt.figure()
    categories = ['Average Ratings', 'Spread of Ratings']
    male_values = [np.mean(male), np.std(male)]
    female_values = [np.mean(fem), np.std(fem)]
    x = range(len(categories))

    plt.bar(x, male_values, width=0.4, label="Males", color='blue', alpha=0.6)
    plt.bar([p + 0.4 for p in x], female_values, width=0.4, label="Females", color='red', alpha=0.6)

    plt.xticks([p + 0.2 for p in x], categories)
    plt.ylabel('Values')
    plt.title('Question 3: Ratings Comparison')
    plt.legend()
    plt.show()
    
    return stat1, p1, stat_2,p_2,d

if __name__ == '__main__':
    num = pd.read_csv('data/rmpCapstoneNum.csv', header=None)
    print(q1(num))