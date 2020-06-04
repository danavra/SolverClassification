import pandas as pd
import os
import warnings
from ModelTesting import leave_one_out, ANSWER_FEATURES, SOLVER_FEATURES
from CombinedModel import combined_leave_one_out
from DataUtil import data_preparation, NOT_ANSWERS
import aggregation_methods as agg

COLUMNS = ['Problem', 'group_num', 'classifier', 'aggregation', 'entity', 'features', 'classification', 'is correct']


# ANSWER_MIN_FEATURES = ANSWER_FEATURES[:-2]
# SOLVER_MIN_FEATURES = SOLVER_FEATURES[:-5]


def baseline(raw_data_dir_path, not_answers=None):
    print('{m} BASELINE : START {m}'.format(m='-' * 100))
    files = [os.path.join(raw_data_dir_path, f) for f in os.listdir(raw_data_dir_path) if f.endswith('.csv')]

    if not files:
        return None

    final_res = {'correct_answer': [], 'Problem': [], 'Classifier': [], 'Classification': [], 'group_number': [],
                 'is_correct': [], 'aggregation': [], 'features': [], 'entity': []}
    for file in files:
        df = pd.read_csv(file)
        if not not_answers:
            not_answers = NOT_ANSWERS
        answers = [col for col in df.columns.values if col not in not_answers]
        df = data_preparation(df, not_answers)
        if 'Confidence' not in df.columns.values:
            print(file.split(os.sep)[-1])
            continue
        for group in df.group_number.unique():
            group_df = df[df.group_number == group]

            try:
                correct_ans = group_df[group_df.Class == 1]['Answer'].values[0]
            except:
                print(group_df.Problem.unique()[0], 'DDDDDD')
            agg_ans = agg.majority_rule(group_df, answers)
            final_res['correct_answer'].append(correct_ans)
            final_res['Problem'].append(group_df['Problem'].unique()[0])
            final_res['Classifier'].append('Majority Rule')
            final_res['Classification'].append(agg_ans)
            final_res['group_number'].append(group)
            final_res['is_correct'].append(int(correct_ans == agg_ans))
            final_res['aggregation'].append('baseline')
            final_res['features'].append('baseline')
            final_res['entity'].append('raw')

            agg_ans = agg.surprisingly_popular(group_df, answers)
            final_res['correct_answer'].append(correct_ans)
            final_res['Problem'].append(group_df['Problem'].unique()[0])
            final_res['Classifier'].append('Surprisingly Popular')
            final_res['Classification'].append(agg_ans)
            final_res['group_number'].append(group)
            final_res['is_correct'].append(int(correct_ans == agg_ans))
            final_res['aggregation'].append('baseline')
            final_res['features'].append('baseline')
            final_res['entity'].append('raw')

            agg_ans = agg.average_confidence(group_df, answers)
            final_res['correct_answer'].append(correct_ans)
            final_res['Problem'].append(group_df['Problem'].unique()[0])
            final_res['Classifier'].append('Average Confidence')
            final_res['Classification'].append(agg_ans)
            final_res['group_number'].append(group)
            final_res['is_correct'].append(int(correct_ans == agg_ans))
            final_res['aggregation'].append('baseline')
            final_res['features'].append('baseline')
            final_res['entity'].append('raw')

            agg_ans = agg.confidence_weighted(group_df, answers)
            final_res['correct_answer'].append(correct_ans)
            final_res['Problem'].append(group_df['Problem'].unique()[0])
            final_res['Classifier'].append('Weighted Confidence')
            final_res['Classification'].append(agg_ans)
            final_res['group_number'].append(group)
            final_res['is_correct'].append(int(correct_ans == agg_ans))
            final_res['aggregation'].append('baseline')
            final_res['features'].append('baseline')
            final_res['entity'].append('raw')

    final_df = pd.DataFrame(data=final_res)
    print('{m} BASELINE : OVER {m}'.format(m='-' * 100))
    return final_df


def add_data_values(df, values):
    for key in values.keys():
        df[key] = values[key]
    return df


def get_normalized_features_df(featured_dir_path, answer_features_file, solver_features_file):
    answer_features_path = os.path.join(featured_dir_path, answer_features_file)
    solver_features_path = os.path.join(featured_dir_path, solver_features_file)

    if not os.path.isfile(answer_features_path) or not os.path.isfile(solver_features_path):
        return None, None

    answer_features = pd.read_csv(answer_features_path)
    answer_features['AvgPS'] = answer_features['AvgPS'].fillna(answer_features['AvgPS'].median())
    answer_features['AvgConf'] = answer_features['AvgConf'].fillna(answer_features['AvgConf'].median())
    solver_features = pd.read_csv(solver_features_path)
    solver_features['cad'] = solver_features['cad'].fillna(solver_features['cad'].median())
    solver_features['cadg'] = solver_features['cadg'].fillna(solver_features['cadg'].median())
    solver_features['Psolve'] = solver_features['Psolve'].fillna(solver_features['Psolve'].median())
    solver_features['arrogance'] = solver_features['arrogance'].fillna(solver_features['arrogance'].median())
    solver_features['Confidence'] = solver_features['Confidence'].fillna(solver_features['Confidence'].median())
    return answer_features, solver_features


def full_data(featured_dir_path, answer_features_file, solver_features_file):
    print('{m} FULL_DATA : START {m}'.format(m='-' * 100))

    answer_features, solver_features = get_normalized_features_df(featured_dir_path, answer_features_file,
                                                                  solver_features_file)

    if answer_features is None or solver_features is None:
        return None

    dfs_list = []
    vals = {'aggregation': 'all_group', 'features': 'minPlus', 'entity': 'answer'}

    kwargs = {'answer': True}
    answer_Conf_group_out_res, answer_Conf_group_out_preds = leave_one_out(answer_features, ANSWER_FEATURES, **kwargs)
    dfs_list.append(add_data_values(answer_Conf_group_out_res, vals))

    solver_Conf_group_out_res, solver_Conf_group_out_preds = leave_one_out(solver_features, SOLVER_FEATURES)
    vals['entity'] = 'solver'
    dfs_list.append(add_data_values(solver_Conf_group_out_res, vals))

    combined_Conf_group_out_res = combined_leave_one_out(answer_features, ANSWER_FEATURES, solver_features,
                                                         SOLVER_FEATURES)
    vals['entity'] = 'combined'
    dfs_list.append(add_data_values(combined_Conf_group_out_res, vals))

    try:
        for key in answer_Conf_group_out_preds.keys():
            pd.concat(answer_Conf_group_out_preds[key]).to_csv(
                os.path.join(os.getcwd(), 'data', 'predictions', 'answer_Conf_group_{}.csv'.format(key)))
            pd.concat(solver_Conf_group_out_preds[key]).to_csv(
                os.path.join(os.getcwd(), 'data', 'predictions', 'solver_Conf_group_{}.csv'.format(key)))
    except Exception as e:
        print(e)

    final_df = pd.concat(dfs_list)
    print('{m} ALL_DATA : OVER {m}'.format(m='-' * 100))
    return final_df


def get_cluster(group, cluster_df):
    clustered = cluster_df[cluster_df.group_number == group]
    res = clustered.cluster.unique()
    clstr = res[0] if len(res) else -2
    return clstr


def clusterwise(featured_dir_path, answer_features_file, solver_features_file, cluster_dir_path):
    print('{m} CLUSTERS : START {m}'.format(m='-' * 100))

    answer_features, solver_features = get_normalized_features_df(featured_dir_path, answer_features_file,
                                                                  solver_features_file)

    if answer_features is None or solver_features is None:
        return None

    files = [os.path.join(cluster_dir_path, f) for f in os.listdir(cluster_dir_path) if f.endswith('.csv')]
    results = []
    for file_path in files:
        method = file_path.split(os.sep)[-1].strip('.csv').strip('cluster_')
        print('{t}{m} Clustering: {met} {m}'.format(t=' ' * 30, m='-' * 70, met=method))

        cluster_df = pd.read_csv(file_path)
        answer_features['cluster'] = answer_features['group_number'].apply(
            lambda x: get_cluster(x, cluster_df))
        solver_features['cluster'] = solver_features['group_number'].apply(
            lambda x: get_cluster(x, cluster_df))

        answer_pred_res = dict()
        solver_pred_res = dict()
        clusters = solver_features.cluster.unique()
        for cluster in clusters:
            print('{t}{m} Cluster: {c} {m}'.format(t=' ' * 50, m='-' * 50, c=cluster))
            answer_cluster_df = answer_features[answer_features.cluster == cluster]
            solver_cluster_df = solver_features[solver_features.cluster == cluster]
            answer_2_add, answer_pred_dict = leave_one_out(answer_cluster_df, ANSWER_FEATURES)
            solver_2_add, solver_pred_dict = leave_one_out(solver_cluster_df, SOLVER_FEATURES)
            ensemble_2_add = combined_leave_one_out(answer_cluster_df, ANSWER_FEATURES, solver_cluster_df,
                                                    SOLVER_FEATURES)
            for key in answer_pred_dict.keys():
                a = pd.concat(answer_pred_dict[key])
                s = pd.concat(solver_pred_dict[key])
                a['cluster'] = cluster
                s['cluster'] = cluster

                if key in answer_pred_res.keys():
                    answer_pred_res[key].append(a)
                    solver_pred_res[key].append(s)
                else:
                    answer_pred_res[key] = [a]
                    solver_pred_res[key] = [s]

            answer_2_add['aggregation'] = 'cluster_{}'.format(method)
            answer_2_add['features'] = cluster
            answer_2_add['entity'] = 'answer'

            solver_2_add['aggregation'] = 'cluster_{}'.format(method)
            solver_2_add['features'] = cluster
            solver_2_add['entity'] = 'solver'

            ensemble_2_add['aggregation'] = 'cluster_{}'.format(method)
            ensemble_2_add['features'] = cluster
            ensemble_2_add['entity'] = 'ensemble'

            results.append(answer_2_add)
            results.append(solver_2_add)
            results.append(ensemble_2_add)

        for key in answer_pred_res.keys():
            pd.concat(answer_pred_res[key]).to_csv(os.path.join(cluster_dir_path, 'results',
                                                                '{method}_{model}_answer.csv'.format(method=method,
                                                                                                     model=key)))
            pd.concat(solver_pred_res[key]).to_csv(os.path.join(cluster_dir_path, 'results',
                                                                '{method}_{model}_solver.csv'.format(method=method,
                                                                                                     model=key)))
    final_df = pd.concat(results)
    print('{m} CLUSTERS : OVER {m}'.format(m='-' * 100))
    return final_df


def run_all_experiments(basline_experiment=True, cluster_experiment=True, full_data_experiment=True):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        features_dir = os.path.join(os.getcwd(), 'data', 'featured data')
        answer_feature_f = 'answer_features.csv'
        solver_feature_f = 'solver_features.csv'
        cluster_dir = os.path.join(os.getcwd(), 'data', 'clustered data')
        raw_data_dir = os.path.join(os.getcwd(), 'data', 'raw data')

        dataframes_list = []

        if basline_experiment:
            res = baseline(raw_data_dir)
            if res is not None:
                dataframes_list.append(res)

        if cluster_experiment:
            res = clusterwise(features_dir, answer_feature_f, solver_feature_f, cluster_dir)
            if res is not None:
                dataframes_list.append(res)

        if full_data_experiment:
            res = full_data(features_dir, answer_feature_f, solver_feature_f)
            if res is not None:
                dataframes_list.append(res)

        exps_results = None
        exps_results_dict = dict()
        models = {'baseline': [('Majority Rule', 'raw'), ('Surprisingly Popular', 'raw'), ('Average Confidence', 'raw'),
                               ('Weighted Confidence', 'raw')],
                  'cluster_dbscan02': [('SVM', 'solver'), ('SVM', 'answer'), ('Vote_soft', 'solver'),
                                       ('Vote_soft', 'answer'), ('Bag_KNN', 'solver'), ('Bag_KNN', 'answer'),
                                       ('Combined_prediction', 'ensemble')],
                  'all_group': [('SVM', 'solver'), ('SVM', 'answer'), ('Vote_soft', 'solver'), ('Vote_soft', 'answer'),
                                ('Bag_KNN', 'solver'), ('Bag_KNN', 'answer'), ('Combined_prediction', 'ensemble')]}
        if dataframes_list:
            exps_results = pd.concat(dataframes_list)
            exps_results.to_csv(os.path.join(os.getcwd(), 'data', 'results.csv'))
            for agg_method in models.keys():
                rel_df = exps_results[exps_results.aggregation == agg_method]
                for modl in models[agg_method]:
                    calc = rel_df[rel_df.Classifier == modl[0]][rel_df.entity == modl[1]]['is_correct']
                    exps_results_dict['{a}_{m}_{e}'.format(a=agg_method, m=modl[0], e=modl[1])] = calc.sum()/len(calc)
        exps_results_dict['path'] = str(os.path.join(os.getcwd(), 'data', 'results.csv'))

        return exps_results_dict


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # run_all_experiments()
