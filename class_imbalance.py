import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks 
from imblearn.under_sampling import NearMiss

df = pd.read_csv("fetal_health.csv")

print(df.head())
print(df.info())
print(df.describe())
#null values
print(df.isna().sum())

#missing values
# Count the missing and null values for dataset fetal healt.
miss_values = df.columns[df.isnull().any()]
print(f"Missing values:\n{df[miss_values].isnull().sum()}")


print(sns.countplot(x='fetal_health', data=df))
plt.show()
print(df['fetal_health'].value_counts())


for i, column in enumerate(df.columns):
    sns.boxplot(x='fetal_health', y=column, data=df)
    plt.title(column)
    plt.show()
    print()

corr = df.corr()

plt.figure(figsize=(24, 20))
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

sns.jointplot(x="accelerations", y="uterine_contractions", data=df, hue="fetal_health")
plt.show()

sns.jointplot(x="prolongued_decelerations", y="uterine_contractions", data=df, hue="fetal_health")
plt.show()

eda_normal = df[df['fetal_health']==1.0]
plt.figure(figsize=(25, 25))
sns.displot(eda_normal, x="prolongued_decelerations", hue="fetal_health", kind="kde", fill=True)
plt.show()

sns.jointplot(x="accelerations", y="abnormal_short_term_variability", data=df, hue="fetal_health")
plt.show()

sns.jointplot(x="percentage_of_time_with_abnormal_long_term_variability", 
	y="abnormal_short_term_variability", data=df, hue="fetal_health")
plt.show()

plt.figure(figsize=(25, 25))
sns.displot(df, x="percentage_of_time_with_abnormal_long_term_variability", hue="fetal_health", kind="kde", fill=True)
plt.show()

plt.figure(figsize=(25, 25))
sns.displot(eda, x="percentage_of_time_with_abnormal_long_term_variability", hue="fetal_health", kind="kde", fill=True)
plt.show()

plt.figure(figsize=(25, 25))
sns.displot(eda_normal, x="percentage_of_time_with_abnormal_long_term_variability", hue="fetal_health", kind="kde", fill=True)
plt.show()

plt.figure(figsize=(25, 25))
sns.displot(df, x="abnormal_short_term_variability", hue="fetal_health", kind="kde", fill=True)
plt.show()

plt.figure(figsize=(25, 25))
sns.displot(df, x="accelerations", hue="fetal_health", kind="kde", fill=True)
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
  print("F1 Score ", f1)
  print("Recall ", recall)
  print("Precision ", precision)
  return [model_name, oversample_type, f1, recall, precision]

models = {
    'DecisionTrees': DecisionTreeClassifier(random_state=42),
    'RandomForest':RandomForestClassifier(random_state=42),
    'LinearSVC':LinearSVC(random_state=0),
    'AdaBoostClassifier':AdaBoostClassifier(random_state=42),
    'SGD':SGDClassifier()
}

sampled_data = {
    'ACTUAL':[X_train, y_train],
    'SMOTE':[X_train_sm, y_train_sm], 
    'TOMEK LINKS':[X_train_tl, y_train_tl],
    'NEAR MISS - 1':[X_train_nm, y_train_nm],
    'NEAR MISS - 2':[X_train_nm2, y_train_nm2],
    'NEAR MISS - 3':[X_train_nm3, y_train_nm3],

}

final_output = []
for model_k, model_clf in models.items():
  for data_type, data in sampled_data.items():
    model_clf.fit(data[0], data[1])
    final_output.append(evaluate_model(model_clf, X_test, y_test, model_k, data_type))

final_df = pd.DataFrame(final_output, columns=['Model', 'DataType', 'F1', 'Recall', 'Precision'])

print(final_df.sort_values(by="F1", ascending=False))