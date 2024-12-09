# -*- coding: utf-8 -*-
"""
@author: sonih
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

numname = {1: "Average Rating",2: "Average Difficulty", 3: "Number of ratings", 4: "Received a pepper?",
           5: "The proportion of students that said they would take the class again",
           6: "The number of ratings coming from online classes",7: "Male", 8: "Female"}
#Map tagnames with the column number
tagname = {1: "Tough grader", 2: "Good feedback", 3: "Respected", 4: "Lots to read", 
           5: "Participation matters", 6: "Don’t skip class or you will not pass", 7: "Lots of homework", 
           8: "Inspirational", 9: "Pop quizzes!", 10: "Accessible", 11: "So many papers", 12: "Clear grading", 
           13: "Hilarious", 14: "Test heavy", 15: "Graded by few things", 16: "Amazing lectures", 
           17: "Caring", 18: "Extra credit", 19: "Group projects", 20: "Lecture heavy"}