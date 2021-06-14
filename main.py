from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, NearMiss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import pre_processing_data
from cost_sensitive_learning import  default_metrics, class_weighting, rebalancing, voting_scores
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Use Seaborn's context settings to make fonts larger.
import seaborn as sns
sns.set_context('talk')

def grouped_bar(loss_arr):

    methods = ['Default', 'Undersampling+Oversampling', 'Class Weighting', 'Costing-Rejection Sampling']
    labels = ['Linear SVC', 'Random Forest', 'Naive Bayes']
    width = 0.8 / len(loss_arr)
    Pos = np.array(range(len(loss_arr[0])))
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = []
    for i in range(len(loss_arr)):
        bars.append(ax.bar(Pos + i * width, loss_arr[i], width=width, label=methods[i]))

    ax.set_xticks(Pos + width / 4)
    ax.set_xticklabels(labels)
    ax.bar_label(bars[0], padding=3)
    ax.bar_label(bars[1], padding=3)
    ax.bar_label(bars[2], padding=3)
    ax.bar_label(bars[3], padding=3)
    ax.legend()
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

    return


if __name__ == '__main__':
    data = pd.read_csv("venv/data/fetal_health.csv")
    X_train, X_test, y_train, y_test = pre_processing_data.pre_processing_binary(data)
    # # explainable(X_train, y_train, X_test,  y_test)
    # default_loss = default_metrics(X_train, y_train, X_test, y_test)
    # under_loss, over_loss, comb_loss =  rebalancing(X_train, y_train, X_test, y_test)
    # class_weighting_loss = class_weighting(X_train, y_train, X_test, y_test)
    # rej_loss = voting_scores(X_train, y_train, X_test, y_test)
    # df2 = pd.DataFrame(np.array([default_loss, comb_loss, class_weighting_loss, rej_loss]))
    # # df2 = df2.T
    # # df2.columns = ['default', 'oversampling', 'class_weighting', 'rejection_sampling']
    # # print(df2)
    # tolist = df2.values.tolist()
    # grouped_bar(tolist)


