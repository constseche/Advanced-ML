import pandas as pd
import numpy as np
from costcla import BayesMinimumRiskClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix,  classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from costcla.sampling import cost_sampling, undersampling


def cost_scores(y_pred, y_test):

    conf_m = confusion_matrix(y_test, y_pred).T
    cost_m = [[0, 4, 5], [1, 0, 1], [1, 1, 0]]
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
    svc_model = SVC(kernel='linear', random_state=0, probability=False, C=1).fit(X_train, y_train, sample_weights)
    y_pred = svc_model.predict(X_test)
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
    c = [1, 5, 6]
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
