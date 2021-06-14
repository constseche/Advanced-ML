import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from costcla.metrics import cost_loss

import cost_sensitive_learning


def explainability(X_train, y_train, X_test, y_val):

    # svc_model = LinearSVC(random_state=0).fit(X_train, y_train)
    # y_pred = svc_model.predict(X_val)
    # print('SVC loss:', cost_score(y_pred, y_val))
    rf_model = RandomForestClassifier(n_estimators = 200, min_samples_split= 10, min_samples_leaf =  1, max_features = 'auto', max_depth =  20, bootstrap =  False, random_state=0).fit(X_train, y_train)

    # ======================== Permutation ===============================

    import eli5
    from eli5.sklearn import PermutationImportance
    from eli5 import explain_prediction

    perm = PermutationImportance(rf_model, random_state=123).fit(X_val, y_val)

    display(eli5.show_weights(perm, feature_names=X_train.columns.tolist(), top=24))

    eli5.show_prediction(rf_model, X_test.iloc[50],
                         feature_names=X_test.columns.tolist(), show_feature_values=True)
    y_pred = rf_model.predict(X_test)
    print('Random Forest loss:', cost_sensitive_learning.cost_scores(y_pred, y_val))

    # ========================= SHAP ===================================

    import shap

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

    shap.summary_plot(shap_values[0], X_test)
    shap.summary_plot(shap_values[1], X_test)
    shap.summary_plot(shap_values[2], X_test)

    return

def pre_processing(data):

    # Split df into X and y
    y = data['fetal_health']
    X = data.drop('fetal_health', axis=1)
    # print(data.describe())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25, stratify=y, random_state=2)

    # StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)


    return X_train, X_test, y_train, y_test


def pre_processing_binary(data):

    y = data['fetal_health']

    for i in range(2126):
        if y[i] == 1.0:
            y[i] = 0.0
        elif y[i] == 3.0:
            y[i] = 1.0
        elif y[i] == 2.0:
            y[i] = 1.0

    X = data.drop('fetal_health', axis=1)
    # print(data.describe())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25, stratify=y, random_state=2)
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    y_test = np.asarray(y_test, dtype=int)
    y_train = np.asarray(y_train, dtype=int)

    return X_train, X_test, y_train, y_test

