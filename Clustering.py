import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from os import getcwd
from os.path import join, isfile
import warnings
from ModelTesting import SOLVER_FEATURES, ANSWER_FEATURES
from run_all_experiments import get_normalized_features_df

META_FEATURES = ['consensus', 'highest_voted_ans', 'variance', 'avg_arrogance', 'med_arrogance', 'var_arrogance',
                 'avg_confidence', 'med_confidence', 'var_confidence', 'avg_EMAM', 'med_EMAM', 'var_EMAM', 'avg_EAAA',
                 'med_EAAA', 'var_EAAA']


def make_cluster(csv_file, clustering_features=[]):
    df = pd.read_csv(csv_file, index_col='group_number')
    if len(clustering_features) != 0:
        df = df[clustering_features]
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(df)
    clust = DBSCAN(eps=0.2).fit(pca_res)
    df['pca1'] = column(pca_res, 0)
    df['pca2'] = column(pca_res, 1)
    df['cluster'] = clust.labels_
    return df


def random_forest(file_name):
    df = pd.read_csv(file_name, index_col='problem')
    df = df.drop(['tnse1','tnse2'],axis=1)
    X = df.drop(['cluster'],axis=1)
    y = df['cluster']
    model = RandomForestClassifier(n_estimators=150)
    model.fit(X,y)
    y_pred = model.predict(X)
    print(accuracy_score(y_pred, y))
    importances = list(model.feature_importances_)
    features = list(X.columns)
    feature_importance = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
    feature_importance = sorted(feature_importance, key=lambda x: x[1])
    [print('Variable {:20} Importance:{}'.format(*pair)) for pair in feature_importance]


def feature_importance(df, features):
    model = RandomForestClassifier(n_estimators=150)
    X = df[features]
    y = df.Class
    model.fit(X, y)
    importances = list(model.feature_importances_)
    res = [(feat, round(imp, 2)) for feat,imp in zip(features, importances)]
    res = sorted(res, key=lambda x: x[1])
    [print('Variable {:20} Importance: {}'.format(*pair)) for pair in res]


def column(matrix, i):
    return [row[i] for row in matrix]


def clustering():
    file_name = join(getcwd(), 'data', 'meta data', 'meta_features.csv')
    if isfile(file_name):
        clustered_df = make_cluster(file_name)
        clustered_df.to_csv(join(getcwd(), 'data', 'clustered data', 'dbscan02.csv'))


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # clustering()
