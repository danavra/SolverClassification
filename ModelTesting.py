import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import os
import warnings
from DataUtil import write_data

ANSWER_FEATURES = ['SD', 'AvgPSAR', 'AvgPSS', 'AvgPSNS', 'PSDAR', 'PSDS', 'PSDNS', 'AvgCS', 'AvgB', 'AvgPS', 'AvgConf']
# SOLVER_FEATURES = ['psma', 'chisqr', 'bs', 'EAMA', 'EAAA', 'ec', 'arrogance', 'cad', 'cadg', 'Confidence', 'Psolve']
SOLVER_FEATURES = ['psma', 'chisqr', 'bs', 'EAMA', 'EAAA', 'ec', 'cad', 'cadg', 'Confidence', 'Psolve']
# ANSWER_FEATURES = ['SD', 'AvgPSAR', 'AvgPSS', 'AvgPSNS', 'PSDAR', 'PSDS', 'PSDNS', 'AvgCS', 'AvgB', 'AvgPS']
# SOLVER_FEATURES = ['psma', 'chisqr', 'bs', 'EAMA', 'ec', 'cad', 'cadg', 'Psolve']


def train_test_split_by_group(df, features, group):
    """
    split the given DataFrame to train and test sets by given group
    :param df: (pandas DataFrame) the given data
    :param features: (list(str)) the features to keep
    :param group: (int) the group for the test set
    :return: features for train, features for test, y for train, y for test
    """
    train = df[df.group_number != group]
    val = df[df.group_number == group]
    X_train = train[features].values
    X_val = val[features].values
    Y_train = train['Class']
    Y_val = val['Class']
    return X_train, X_val, Y_train, Y_val


def train_test_split_by_problem(df, features, group):
    """
    split the given DataFrame to train and test sets by given group's problem
    :param df: (pandas DataFrame) the given data
    :param features: (list(str)) the features to keep
    :param group: (int) the group for the test set
    :return: features for train, features for test, y for train, y for test
    """
    val = df[df.group_number == group]
    prob = val.Problem.unique()[0]
    train = df[df.Problem != prob]
    X_train = train[features].values
    X_val = val[features].values
    Y_train = train['Class']
    Y_val = val['Class']
    return X_train, X_val, Y_train, Y_val


def get_models(**kwargs):
    models = dict()
    # models['LR'] = LogisticRegression(max_iter=2000, random_state=40, class_weight={True: 2, False: 1})
    if (kwargs.get('answer') and kwargs['answer']) or (kwargs.get('solver') and not kwargs['solver']):
        # models['RF'] = RandomForestClassifier(n_estimators=250, max_depth=2, min_samples_leaf=6, class_weight={True: 2, False: 1})
        voters = [('LDA', LinearDiscriminantAnalysis()),
                  ('RF', RandomForestClassifier(n_estimators=250, max_depth=2, min_samples_leaf=6,
                                                class_weight={True: 2, False: 1})),
                  ('LR', LogisticRegression(max_iter=2000, random_state=40, class_weight={True: 2, False: 1})),
                  ('XGB', XGBClassifier()),
                  ('KNN', KNeighborsClassifier(n_neighbors=7, leaf_size=20))]
    else:
        # models['RF'] = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=4, class_weight={True: 2, False: 1})
        voters = [('LDA', LinearDiscriminantAnalysis()),
                  ('RF', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=4,
                                                class_weight={True: 2, False: 1})),
                  ('LR', LogisticRegression(max_iter=2000, random_state=40, class_weight={True: 2, False: 1})),
                  ('XGB', XGBClassifier()),
                  ('KNN', KNeighborsClassifier(n_neighbors=7, leaf_size=20))]
    # models['LDA'] = LinearDiscriminantAnalysis()
    models['SVM'] = SVC(probability=True)
    # models['KNN'] = KNeighborsClassifier(n_neighbors=7, leaf_size=20)
    # models['XGB'] = XGBClassifier()
    # models['Stack'] = StackingClassifier(estimators=voters, final_estimator=XGBClassifier())
    models['Vote_soft'] = VotingClassifier(estimators=voters, voting='soft')
    models['Bag_KNN'] = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=7, leaf_size=20), n_estimators=17)
    return models


def get_models_classification(train_set, test_set, features, **kwargs):
    models = get_models(**kwargs)
    res = dict()
    X_train = train_set[features]
    y_train = train_set.Class
    X_test = test_set[features]
    predictions_dict = dict()
    for model in models.keys():
        models[model].fit(X_train, y_train)
        predictions = models[model].predict(X_test)
        test_set['pred'] = predictions
        probas = models[model].predict_proba(X_test)
        probas = list(map(lambda x: x[1], probas))
        test_set['probas'] = probas
        predictions_dict[model] = pd.DataFrame.copy(test_set)
        model_ans = None
        if len(test_set[test_set.pred == True]) != 0:
            ans_count = test_set[test_set.pred == True]['Answer'].value_counts()
            if len(ans_count.index) == 1 or (len(ans_count.index) > 1 and ans_count.values[0] > ans_count.values[1]):
                model_ans = ans_count.index[0]

        if model_ans is None:
            model_ans = test_set[test_set.probas == test_set.probas.max()]['Answer'].values[0]

        res[model] = model_ans

    return res, predictions_dict


def get_models_classification_proba(train_set, test_set, features):
    models = get_models()
    res = dict()
    X_train = train_set[features]
    y_train = train_set.Class
    X_test = test_set[features]
    for model in models.keys():
        models[model].fit(X_train, y_train)
        predictions = models[model].predict_proba(X_test)
        predictions = list(map(lambda x: x[1], predictions))
        test_set['probas'] = predictions
        res[model] = test_set[test_set.probas == test_set.probas.max()]['Answer'].values[0]
    return res


def leave_one_out(df, features, bool_problem=False, **kwargs):
    final_res = {'correct_answer': [], 'Problem': [], 'Classifier': [], 'Classification': [], 'group_number': [],
                 'is_correct': []}
    predictions_dict = {}
    groups = df.group_number.unique()
    for group in groups:
        print('group {num} out of {all}'.format(num=group, all=len(groups)))
        problem = df[df.group_number == group].Problem.unique()[0]
        if bool_problem:
            train_df = df[df.Problem != problem]
        else:
            train_df = df[df.group_number != group]

        test_df = df[df.group_number == group]
        train_nulls_dict = dict()
        test_nulls_dict = dict()
        for feat in features:
            train_nulls_dict[feat] = list(train_df[train_df[feat].isnull()].index)
            test_nulls_dict[feat] = list(test_df[test_df[feat].isnull()].index)
        res, pred_dfs_dict = get_models_classification(train_df, test_df, features, **kwargs)

        for key in res.keys():
            if key in predictions_dict.keys():
                predictions_dict[key].append(pred_dfs_dict[key])
            else:
                predictions_dict[key] = [pred_dfs_dict[key]]

            final_res['Problem'].append(problem)
            final_res['group_number'].append(group)
            correct = test_df[test_df.Class == 1].iloc[0]['Answer']
            final_res['correct_answer'].append(correct)
            final_res['Classifier'].append(key)
            final_res['Classification'].append(res[key])

            if res[key] == correct:
                final_res['is_correct'].append(1)
            else:
                final_res['is_correct'].append(0)

    results = pd.DataFrame(data=final_res)
    return results, predictions_dict


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        print('ModelTesting')