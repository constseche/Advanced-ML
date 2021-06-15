import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, balanced_accuracy_score, roc_auc_score
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks 
from imblearn.under_sampling import NearMiss
import numpy as np
import math
from sklearn.metrics import roc_curve,auc

df = pd.read_csv("fetal_health.csv")

print(df.head())
print(df.info())
print(df.describe())
#null values
print(df.isna().sum())

#missing values
miss_values = df.columns[df.isnull().any()]
print(f"Missing values:\n{df[miss_values].isnull().sum()}")


print(sns.countplot(x='fetal_health', data=df))
plt.show()
print(df['fetal_health'].value_counts())


for i, column in enumerate(df.columns):
    sns.boxplot(x='fetal_health', y=column, data=df)
    plt.title(column)
    plt.show()
    #print()
    

corr = df.corr()
plt.figure(figsize=(24, 20))
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()


print("There are total "+str(len(df))+" rows in the dataset")

X = df.drop(["fetal_health"],axis=1)
Y = df["fetal_health"]

#Step by Step "Fetal Health" Prediction-Detailed - ekshghsh gia standard scaler
std_scale = StandardScaler()
X_sc = std_scale.fit_transform(X)


X_train, X_test, y_train,y_test = train_test_split(X_sc, Y, test_size=0.25, random_state=42)
print("There are total "+str(len(X_train))+" rows in training dataset")
print("There are total "+str(len(X_test))+" rows in test dataset")

smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

tl = TomekLinks()
X_train_tl, y_train_tl = tl.fit_resample(X_train, y_train)

nm = NearMiss(version = 1)
X_train_nm, y_train_nm = nm.fit_resample(X_train, y_train)
nm2 = NearMiss(version = 2)
X_train_nm2, y_train_nm2 = nm2.fit_resample(X_train, y_train)
nm3 = NearMiss(version = 3)
X_train_nm3, y_train_nm3 = nm3.fit_resample(X_train, y_train)



def evaluate_model(clf, X_test, y_test, model_name, oversample_type):
  print('--------------------------------------------')
  print('Model ', model_name)
  print('Data Type ', oversample_type)
  y_pred = clf.predict(X_test)

  f1 = f1_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')
  precision = precision_score(y_test, y_pred, average='weighted')
  print(classification_report(y_test, y_pred))
  balanced = balanced_accuracy_score(y_test, y_pred)
 
     

  f1= round(f1,2)
  recall= round(recall,2)
  precision= round(precision,2)
  balanced= round(balanced,2)
  


  print("F1 Score",f1)
  print("Recall",recall)
  print("Precision", precision)
  print("Balanced Accuracy Score", balanced)
  roc=0
  if ( model_name == 'DecisionTrees' or model_name == 'RandomForest' or model_name == 'AdaBoostClassifier') :
    y_prob = clf.predict_proba(X_test)
    roc = roc_auc_score(y_test, y_prob, multi_class="ovo",  average="macro")
    print("ROC", roc)
    roc= round(roc,2)

    
  
  
  
  
  return [model_name, oversample_type, f1, recall, precision, balanced, roc]

models = {
    'DecisionTrees': DecisionTreeClassifier(random_state=42),
    'RandomForest':RandomForestClassifier(random_state=42),
    'LinearSVC':LinearSVC(random_state=0),
    'AdaBoostClassifier':AdaBoostClassifier(random_state=42),
    'SGD':SGDClassifier()
}

sampled_data = {
    'Default':[X_train, y_train],
    'SMOTE':[X_train_sm, y_train_sm], 
    'Tomek Links':[X_train_tl, y_train_tl],
    'Near Miss 1':[X_train_nm, y_train_nm],
    'Near Miss 2':[X_train_nm2, y_train_nm2],
    'Near Miss 3':[X_train_nm3, y_train_nm3]

}

final_output = []
for model_k, model_clf in models.items():
  for data_type, data in sampled_data.items():
    model_clf.fit(data[0], data[1])
    final_output.append(evaluate_model(model_clf, X_test, y_test, model_k, data_type))

final_df = pd.DataFrame(final_output, columns=['Model', 'DataType', 'F1', 'Recall','Precision','Balanced Accuracy', 'ROC'])
print(final_df)

def grouped_bar_imbalance(loss_arr):
	methods = ['Default', 'Tomek Links', 'SMOTE', 'Near Miss1']
	labels = ['Random Forest', 'Linear SVC', 'AdaBoostClassifier', 'DecisionTrees', 'SGD']

	width = 0.8 / len(loss_arr)
	Pos = np.array(range(len(loss_arr[0])))
	fig, ax = plt.subplots(figsize=(12, 8))
	bars = []
	for i in range(len(loss_arr)):
		bars.append(ax.bar(Pos + i * width, loss_arr[i], width=width, label=methods[i]))

	ax.set_xticks(Pos + width / 4)
	ax.set_xticklabels(labels)
	ax.bar_label(bars[0], padding=1)
	ax.bar_label(bars[1], padding=1)
	ax.bar_label(bars[2], padding=1)
	ax.bar_label(bars[3], padding=1)
	ax.legend()
	fig.tight_layout()
	plt.show()
	return

for j in range(2,7):
	rf_f1 = []
	for i in range(6, 10):
	    rf_f1.append(final_output[i][j])

	svc_f1 = []
	for i in range(12, 16):
		svc_f1.append(final_output[i][j])

	ada_f1 = []
	for i in range(18, 22):
		ada_f1.append(final_output[i][j])

	dts_f1 = []
	for i in range(0,4):
		dts_f1.append(final_output[i][j])

	sgd_f1 = []
	for i in range(24,28):
		sgd_f1.append(final_output[i][j])

	df2 = pd.DataFrame(np.array([rf_f1, svc_f1, ada_f1, dts_f1, sgd_f1]))
	df2 = df2.T

	tolist = df2.values.tolist()
	#print(tolist)
	#print(len(tolist))
	grouped_bar_imbalance(tolist)
