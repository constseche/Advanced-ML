import pandas as pd

# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# list(data.target_names) # 0 is malignant, 1 is benign
# frame = pd.DataFrame(data.target, columns=['target'])
# frame['target'].value_counts()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from costcla.metrics import cost_loss

import cost_sensitive_learning

data = pd.read_csv("venv/data/fetal_health.csv")
X_train, X_test, y_train, y_test = cost_sensitive_learning.pre_processing(data)

#fp, fn, tp, tn
# create an example-dependent cost-matrix required by costclas
fp = np.full((y_test.shape[0],1), 1)
fn = np.full((y_test.shape[0],1), 4)
tp = np.zeros((y_test.shape[0],1))
tn = np.zeros((y_test.shape[0],1))
cost_matrix = np.hstack((fp, fn, tp, tn))

# create a classic cost-matrix
cost_m = [[0 , 4], [1, 0]]

names = ['random forest', 'linear SVM']
classifiers = [RandomForestClassifier(n_estimators=100, random_state=0),
               SVC(kernel='linear', C=1)]

for name, clf in zip(names, classifiers):
  print(name)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(classification_report(y_test, y_pred, target_names=data.target_names))

  conf_m = confusion_matrix(y_test, y_pred).T # transpose to align with slides
  print(conf_m)
  print(np.sum(conf_m * cost_m))
  loss = cost_loss(y_test, y_pred, cost_matrix)
  print("%d\n" %loss)

"""Minimizing the expected cost"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from costcla.models import BayesMinimumRiskClassifier

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
# 0 is malignant, 1 is benign
#fp, fn, tp, tn
fp = np.full((y_test.shape[0],1), 4)
fn = np.full((y_test.shape[0],1), 1)
tp = np.zeros((y_test.shape[0],1))
tn = np.zeros((y_test.shape[0],1))
cost_matrix = np.hstack((fp, fn, tp, tn))

print("no cost minimization")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
model = clf.fit(X_train, y_train)
pred_test = model.predict(X_test)
print(classification_report(y_test, pred_test, target_names=data.target_names))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides


print("no calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
model = clf.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test, target_names=data.target_names))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("costcla calibration on training set")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
model = clf.fit(X_train, y_train)
prob_train = model.predict_proba(X_train)
bmr = BayesMinimumRiskClassifier(calibration=True)
bmr.fit(y_train, prob_train)
prob_test = model.predict_proba(X_test)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test, target_names=data.target_names))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("\nsigmoid calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
cc = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test, target_names=data.target_names))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("\nisotonic calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
cc = CalibratedClassifierCV(clf, method="isotonic", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test, target_names=data.target_names))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from costcla.metrics import cost_loss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)

# 0 is malignant, 1 is benign
#fp, fn, tp, tn
fp = np.full((y_test.shape[0],1), 4)
fn = np.full((y_test.shape[0],1), 1)
tp = np.zeros((y_test.shape[0],1))
tn = np.zeros((y_test.shape[0],1))
cost_matrix = np.hstack((fp, fn, tp, tn))


clf = RandomForestClassifier(n_estimators=100, random_state=0)
print("without sampling")
print(Counter(y_train))
#0: 149, 1: 249

model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=data.target_names))
print(confusion_matrix(y_test, y_pred).T) # transpose to align with slides
loss = cost_loss(y_test, y_pred, cost_matrix)
print("%d\n" %loss)

print("with undersampling")
sampler = RandomUnderSampler(sampling_strategy={0: 149, 1: 37}, random_state=1)
X_rs, y_rs = sampler.fit_resample(X_train, y_train)
print(Counter(y_rs))

model = clf.fit(X_rs, y_rs)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=data.target_names))
print(confusion_matrix(y_test, y_pred).T) # transpose to align with slides
loss = cost_loss(y_test, y_pred, cost_matrix)
print("%d\n" %loss)

print("with oversampling")
sampler = RandomOverSampler(sampling_strategy={0: 1000, 1: 249}, random_state=1)
X_rs, y_rs = sampler.fit_resample(X_train, y_train)
print(Counter(y_rs))

model = clf.fit(X_rs, y_rs)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=data.target_names))
print(confusion_matrix(y_test, y_pred).T) # transpose to align with slides
loss = cost_loss(y_test, y_pred, cost_matrix)
print("%d\n" %loss)

print("with combination")
sampler = RandomUnderSampler(sampling_strategy={0: 149, 1: 100}, random_state=1)
X_rs, y_rs = sampler.fit_resample(X_train, y_train)
sampler = RandomOverSampler(sampling_strategy={0: 400, 1: 100}, random_state=1)
X_rs, y_rs = sampler.fit_resample(X_rs, y_rs)
print(Counter(y_rs))

model = clf.fit(X_rs, y_rs)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=data.target_names))
print(confusion_matrix(y_test, y_pred).T) # transpose to align with slides
loss = cost_loss(y_test, y_pred, cost_matrix)
print("%d\n" %loss)

"""Using sample weights"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from costcla.metrics import cost_loss

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)

# 0 is malignant, 1 is benign
#fp, fn, tp, tn
fp = np.full((y_test.shape[0],1), 4)
fn = np.full((y_test.shape[0],1), 1)
tp = np.zeros((y_test.shape[0],1))
tn = np.zeros((y_test.shape[0],1))
cost_matrix = np.hstack((fp, fn, tp, tn))

print("without weights")
clf = RandomForestClassifier(n_estimators=10, random_state=0)
#clf = SVC(kernel='linear', probability=False, C=1)
#clf = DecisionTreeClassifier()
model = clf.fit(X_train, y_train)
pred_test = model.predict(X_test)

print(classification_report(y_test, pred_test, target_names=data.target_names))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("\nwith weights")
# now create the sample weights according to y
weights = np.zeros(y_train.shape[0])
weights[np.where(y_train == 1)] = 1;
weights[np.where(y_train == 0)] = 4;
#print(data.DESCR)

model = clf.fit(X_train, y_train, weights)
pred_test = clf.predict(X_test)

print(classification_report(y_test, pred_test, target_names=data.target_names))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("\nwith weights (alternative)")
clf = RandomForestClassifier(n_estimators=10, random_state=0, class_weight={0: 4, 1: 1})
model = clf.fit(X_train, y_train)
pred_test = model.predict(X_test)

print(classification_report(y_test, pred_test, target_names=data.target_names))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

"""Another probability calibration example"""

import time
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier

data = datasets.load_breast_cancer()
#data = datasets.load_iris()
#data = datasets.load_digits()


classifiers = []

ada = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=100, random_state=1)
classifiers.append([ada, "AdaBoost-ed tree"])

ada_cal = CalibratedClassifierCV(AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=100, random_state=1), cv=2, method='isotonic')
classifiers.append([ada_cal, "calibrated AdaBoost-ed tree (isotonic)"])

ada_cal2 = CalibratedClassifierCV(AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=100, random_state=1), cv=2, method='sigmoid')
classifiers.append([ada_cal2, "calibrated AdaBoost-ed tree (sigmoid)"])

neighbors = 10;
knn = KNeighborsClassifier(n_neighbors=neighbors)
classifiers.append([knn, "kNN"])

knn_cal = CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=neighbors), cv=2, method='isotonic')
classifiers.append([knn_cal, "calibrated kNN (isotonic)"])

knn_cal2 = CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=neighbors), cv=2, method='sigmoid')
classifiers.append([knn_cal2, "calibrated kNN (sigmoid)"])


for classifier, label in classifiers:
    start = time.time()
    scores = cross_val_score(classifier, data.data, data.target, cv=10, scoring="neg_log_loss")
    stop = time.time()
    print("%20s neg log_loss: %0.2f (+/- %0.2f), time:%.4f" % (label, scores.mean(), scores.std() * 2, stop - start))

print()

for classifier, label in classifiers:
    start = time.time()
    scores = cross_val_score(classifier, data.data, data.target, cv=10, scoring="neg_mean_squared_error")
    stop = time.time()
    print("%20s squared error: %0.2f (+/- %0.2f), time:%.4f" % (label, scores.mean(), scores.std() * 2, stop - start))

