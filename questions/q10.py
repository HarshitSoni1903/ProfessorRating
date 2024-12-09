# -*- coding: utf-8 -*-
"""
@author: sonih
"""

from imports import pd, plt
from imports import LogisticRegression, train_test_split, StandardScaler
from imports import classification_report, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
from imports import SMOTE


def q10(num,tags):
    pepper = pd.concat([num, tags], axis=1, ignore_index=True)
    pepper.dropna(inplace=True)    
    x_pepp = pepper.drop(3, axis=1)
    y_pepp = pepper[3]
    X_train, X_test, y_train, y_test = train_test_split(x_pepp, y_pepp, test_size=0.2, random_state=17865635)
    smote = SMOTE(random_state = 17865635)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    le = LogisticRegression(random_state=17865635)
    le.fit(X_train,y_train)
    pred = le.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, pred)    
    mt = confusion_matrix(y_test, pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Question 10")
    roc_auc = roc_auc_score(y_test, pred)
    axes[0].plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    axes[0].plot([0, 1], [0, 1], color='red', linestyle='--')
    axes[0].set_title('ROC Curve')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')

    disp = ConfusionMatrixDisplay(confusion_matrix=mt, display_labels=le.classes_)
    disp.plot(ax=axes[1], cmap='Blues', colorbar=False)
    axes[1].set_title('Confusion Matrix')    
    plt.tight_layout()
    plt.show()  
    return roc_auc_score(y_test, pred), classification_report(y_test, pred)

if __name__ == '__main__':
    num = pd.read_csv('data/rmpCapstoneNum.csv', header=None)
    tags = pd.read_csv('data/rmpCapstoneTags.csv', header=None)
    q10(num,tags)