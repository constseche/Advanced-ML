import pandas as pd
import numpy as np
import sns as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.core.display import display
from sklearn.metrics import plot_confusion_matrix, confusion_matrix,  classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from collections import Counter
from sklearn.preprocessing import StandardScaler
from costcla.sampling import cost_sampling, undersampling
import seaborn as sns
sns.set_context('talk')
import warnings
warnings.simplefilter("ignore", UserWarning)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler



def printCfm(df_cfm):
    sns.heatmap(df_cfm, annot=True, annot_kws={"size": 15}, fmt="d")
    plt.show()

    return


def explainability(X_train, y_train, X_test, y_val):

    rf_model = RandomForestClassifier(n_estimators = 200, min_samples_split= 10, min_samples_leaf =  1, max_features = 'auto', max_depth =  20, bootstrap =  False, random_state=0).fit(X_train, y_train)

    # ======================== Permutation ===============================

    import eli5
    from eli5.sklearn import PermutationImportance
    from eli5 import explain_prediction

    perm = PermutationImportance(rf_model, random_state=123).fit(X_test, y_val)

    display(eli5.show_weights(perm, feature_names=X_train.columns.tolist(), top=24))

    eli5.show_prediction(rf_model, X_test.iloc[50],
                         feature_names=X_test.columns.tolist(), show_feature_values=True)
    y_pred = rf_model.predict(X_test)
    print('Random Forest loss:', cost_scores(y_pred, y_val))

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

def cost_scores(y_pred, y_test):
    conf_m = confusion_matrix(y_test, y_pred)
    printCfm(conf_m)
    conf_m = confusion_matrix(y_test, y_pred).T
    cost_m = [[0, 4, 5], [1, 0, 1], [1, 1, 0]]
    printCfm(conf_m)
    # disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues)
    loss   = np.sum(conf_m * np.array(cost_m))

    print(conf_m)
    print(classification_report(y_test, y_pred))

    return loss


def default_metrics(X_train, y_train, X_test, y_test):
    ret_loss = []

    svc_model = LinearSVC(random_state=0).fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    loss = cost_scores(y_pred, y_test)
    ret_loss.append(loss)
    print('SVC loss:', loss)

    rf_model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    loss = cost_scores(y_pred, y_test)
    ret_loss.append(loss)
    print('Random Forest loss:', loss)

    nb_model = GaussianNB().fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    loss = cost_scores(y_pred, y_test)
    ret_loss.append(loss)
    print('Naive Bayes loss:', loss)
    print(ret_loss)

    return ret_loss


def class_weighting(X_train, y_train, X_test, y_test):
    ret_loss = []

    sample_weights = []
    for y in y_train:
        if y == 1:
            sample_weights.append(1.)
        elif y == 2:
            sample_weights.append(5.)
        elif y == 3:
            sample_weights.append(6.)

    print("==========SVC with class weighting==========")
    # class_weight = {1: 2, 2: 5, 3: 6}
    svc_model = SVC(kernel='linear', random_state=0, probability=False, C=1).fit(X_train, y_train, sample_weights)
    plot_confusion_matrix(svc_model, X_test, y_test, cmap=plt.cm.Blues)
    y_pred = svc_model.predict(X_test)
    # plot_confusion_matrix(svc_model, X_test, y_test, cmap=plt.cm.Blues)

    plt.show()

    loss = cost_scores(y_pred, y_test)
    ret_loss.append(loss)
    print('loss:', loss)

    print("==========Random Forest with class weighting==========")
    rf_model = RandomForestClassifier(random_state=0, class_weight={1: 1, 2: 5, 3: 6}).fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    loss = cost_scores(y_pred, y_test)
    ret_loss.append(loss)
    print('loss:', loss)

    print("==========Naive Bayes with class weighting==========")
    sample_weights = []
    for y in y_train:
        if y == 1:
            sample_weights.append(1.)
        elif y == 2:
            sample_weights.append(5.)
        elif y == 3:
            sample_weights.append(6.)

    nb_model = GaussianNB().fit(X_train, y_train,sample_weight=sample_weights)
    y_pred = nb_model.predict(X_test)
    loss = cost_scores(y_pred, y_test)
    ret_loss.append(loss)
    print('loss:', loss)

    return ret_loss


def rebalancing(X_train, y_train, X_test, y_test):
    names = ['Linear SVM', 'Random forest', 'Naive Bayes']
    clfs = []
    clfs.append(LinearSVC(random_state=0))
    clfs.append(RandomForestClassifier(random_state=0, n_estimators=100, max_depth=70))
    clfs.append(GaussianNB())

    under_loss = []
    for clf, n in zip(clfs, names):
        print("==========Undersampling==========")
        sampler = RandomUnderSampler(sampling_strategy={1: 200, 2: 221, 3: 132}, random_state=0)
        X_rs, y_rs = sampler.fit_resample(X_train, y_train)
        print(Counter(y_rs))
        model = clf.fit(X_rs, y_rs)
        y_pred = clf.predict(X_test)
        loss = cost_scores(y_pred, y_test)
        under_loss.append(loss)
        print("%s" %n, "%d\n" %loss )
    #
    #
    over_loss = []
    for clf, n in zip(clfs, names):
        print("==========Oversampling==========")
        sampler = RandomOverSampler(sampling_strategy={1: 1241, 2: 1000, 3: 1200}, random_state=0)
        X_rs, y_rs = sampler.fit_resample(X_train, y_train)
        print(Counter(y_rs))
        model = clf.fit(X_rs, y_rs)
        y_pred = clf.predict(X_test)
        loss = cost_scores(y_pred, y_test)
        over_loss.append(loss)
        print("%s" %n, "%d\n" %loss)

    comb_loss = []
    count_y = Counter(y_train)
    major_class = count_y[1]
    minor_1_class = count_y[2]
    minor_2_class = count_y[3]
    c = [2, 5, 6]
    cost_major = int(major_class/c[1])
    cost_minor1 = int(major_class/c[0])
    cost_minor2 = int(major_class / c[0])

    for clf, n in zip(clfs, names):
        print("==========Combination============")
        sampler = RandomUnderSampler(sampling_strategy={1: 200, 2: 221, 3: 132}, random_state=0)
        X_rs, y_rs = sampler.fit_resample(X_train, y_train)
        sampler = RandomOverSampler(sampling_strategy={1: 200, 2: 1000, 3: 1200}, random_state=0)
        X_rs, y_rs = sampler.fit_resample(X_rs, y_rs)
        print(Counter(y_rs))
        model = clf.fit(X_rs, y_rs)
        y_pred = clf.predict(X_test)
        loss =  cost_scores(y_pred, y_test)
        comb_loss.append(loss)
        print("%s" %n,"%d\n" %loss)


    return under_loss, over_loss, comb_loss


def rejection_sampling(X_train, y_train):

    c = [2., 5., 6.]
    zeta = 6.
    X_sample = []
    y_sample = []

    for X, y in zip(X_train.values, y_train.values):
        if y == 1:
            prob = c[0] / zeta
        elif y == 2:
            prob = c[1] / zeta
        elif y == 3:
            prob = c[2] / zeta

        sample_item = np.random.choice([True, False], p = [prob, 1 - prob])

        if sample_item:
            X_sample.append(X)
            y_sample.append(y)

    X_sample = np.array(X_sample)
    y_sample = np.array(y_sample)

    return X_sample, y_sample

def hard_votting(clfs, X_val):

    # ensemble = VotingClassifier(estimators=clfs, voting='hard')
    # ensemble.fit(X_train, y_train)
    # y_pred = ensemble.predict(X_test)

    # ============ from sklearn.ensemble.VotingClassifier =================
    # predictions = self._predict(X)
    # maj = np.apply_along_axis(
    #     lambda x: np.argmax(
    #         np.bincount(x, weights=self._weights_not_none)),
    #     axis=1, arr=predictions)

    y_pred = np.asarray([clf.predict(X_val) for clf in clfs]).T
    y_pred = np.apply_along_axis(lambda x:
                               np.argmax(np.bincount(x)), axis=1, arr=y_pred.astype('int')
                               )

    return y_pred


def voting_scores(X_train, y_train, X_test, y_test):
    rej_loss = []
    svc_models = []
    rf_models = []
    nb_models = []

    for i in range(10):
        X_train_sample, y_train_sample = rejection_sampling(X_train, y_train)
        svc_models.append(LinearSVC(random_state=0).fit(X_train_sample, y_train_sample))
        rf_models.append(RandomForestClassifier(random_state=0).fit(X_train_sample, y_train_sample))
        nb_models.append(GaussianNB().fit(X_train_sample, y_train_sample))

    # X_train_rej_sample, y_train_rej_sample = rejection_sampling(X_train, y_train)
    # print(X_train_rej_sample.shape)
    # print(y_train_rej_sample.shape)

    print("==========SVC==========")
    # svc_clf = LinearSVC(random_state=0).fit(X_train_rej_sample, y_train_rej_sample)
    # y_pred  = svc_clf.predict(X_test)
    y_pred = hard_votting(svc_models, X_test)
    loss = cost_scores(y_pred, y_test)
    rej_loss.append(loss)
    print('SVC with rejection sampling-votting:', loss)

    print("==========Random Forest==========")
    # rf_clf = RandomForestClassifier(random_state=0).fit(X_train_rej_sample, y_train_rej_sample)
    # y_pred = rf_clf.predict(X_test)
    y_pred = hard_votting(rf_models, X_test)
    loss = cost_scores(y_pred, y_test)
    rej_loss.append(loss)
    print('Random Forest with rejection sampling-votting:', loss)

    print("==========Naive Bayes==========")
    # nb_clf = GaussianNB().fit(X_train_rej_sample, y_train_rej_sample)
    # y_pred = nb_clf.predict(X_test)
    y_pred = hard_votting(nb_models, X_test)
    loss = cost_scores(y_pred, y_test)
    rej_loss.append(loss)
    print('Naive Bayes with rejection sampling:', loss)

    return rej_loss


# def run():
data = pd.read_csv("/Users/conatseche/PycharmProjects/Advanced-ML/data/fetal_health.csv")
X_train, X_test, y_train, y_test = pre_processing(data)
# # explainable(X_train, y_train, X_test,  y_test)
default_loss = default_metrics(X_train, y_train, X_test, y_test)
under_loss=  rebalancing(X_train, y_train, X_test, y_test)
class_weighting_loss = class_weighting(X_train, y_train, X_test, y_test)
rej_loss = voting_scores(X_train, y_train, X_test, y_test)
df2 = pd.DataFrame(np.array([default_loss, under_loss, class_weighting_loss, rej_loss]))
# df2 = df2.T
# df2.columns = ['default', 'oversampling', 'class_weighting', 'rejection_sampling']
# print(df2)8
tolist = df2.values.tolist()
grouped_bar(tolist)
