import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scikitplot.metrics import plot_roc
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.ensemble import VotingClassifier
from tqdm import tqdm


def voting_classifier(preds):
    from collections import Counter

    final_pred = []
    n = len(preds[0])
    k = len(preds)
    for i in range(n):
        nth_pred = []
        for j in range(k):
            y_pred = preds[j][i]
            nth_pred.append(y_pred)
        counter = Counter(nth_pred)
        most_common = counter.most_common(1)[0]
        final_pred.append(most_common[0])
    return final_pred


def bagging(x_train, y_train, x_test, y_test, base_estimator, n_estimators, estimator_params):
    size = len(x_train)

    acc, f1 = [], []
    for n in tqdm(n_estimators):
        pred = []
        for i in range(n):
            x_bts, y_bts = resample(x_train, y_train, replace=True, n_samples=size)
            model = base_estimator(**estimator_params)
            model.fit(x_bts, y_bts)
            model_pred = model.predict(x_test)
            pred.append(model_pred)

        ensemble_pred = voting_classifier(pred)
        acc.append(accuracy_score(y_test, ensemble_pred))
        f1.append(f1_score(y_test, ensemble_pred, average='macro'))

    return acc, f1


dataset = pd.read_csv('car_data.csv', delimiter=',', header=None).values


def train_model(x_train, y_train, x_test, y_test, base_estimator, estimator_params):
    model = base_estimator(**estimator_params)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    f1 = f1_score(y_test, model.predict(x_test), average='macro')
    return acc, f1


if __name__ == '__main__':
    data = dataset[:, :-1].astype(int)
    target = dataset[:, -1]

    print(data.shape, target.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.35)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    N = len(X_train)

    log_reg_params = {'C': 1.0,
                      'fit_intercept': True,
                      'max_iter': 500,
                      'multi_class': 'multinomial',
                      'penalty': 'l1',
                      'solver': 'saga',
                      'tol': 0.001}

    dt_params = {'max_depth': 2}

    rf_params = {'criterion': 'gini',
                 'max_depth': 2,
                 'max_features': 'sqrt',
                 'min_samples_leaf': 1,
                 'min_samples_split': 2}

    params = [
        (X_train_std, y_train, X_test_std, y_test, LogisticRegression, log_reg_params),
        (X_train, y_train, X_test, y_test, DecisionTreeClassifier, dt_params),
        (X_train, y_train, X_test, y_test, RandomForestClassifier, rf_params),
    ]

    res = []
    for part in reversed(range(1, 11)):
        n = N // part
        part_results = []
        for param in params:
            x_train, y_train, x_test, y_test, model_type, model_params = param
            x_train_bts, y_train_bts = resample(x_train, y_train, replace=True, n_samples=part, stratify=y_train)
            part_res = {}
            acc, f1 = train_model(x_train_bts, y_train_bts, x_test, y_test, model_type, model_params)
            part_res["model_name"] = model_type.__name__
            part_res["acc"] = acc
            part_res["f1"] = f1
            part_res["part"] = part
            part_results.append(part_res)
        res.append(part_results)
