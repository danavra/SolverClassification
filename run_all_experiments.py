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
    print('{m} BASELINE : START {m}'.format(m='-'*100))
    files = [os.path.join(raw_data_dir_path, f) for f in os.listdir(raw_data_dir_path) if f.endswith('.csv')]
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
    print('{m} BASELINE : OVER {m}'.format(m='-'*100))
    return final_df


def add_data_values(df, values):
    for key in values.keys():
        df[key] = values[key]
    return df


def get_normalized_features_df(featured_dir_path, answer_features_file, solver_features_file):
    answer_features = pd.read_csv(os.path.join(featured_dir_path, answer_features_file))
    answer_features['AvgPS'] = answer_features['AvgPS'].fillna(answer_features['AvgPS'].median())
    answer_features['AvgConf'] = answer_features['AvgConf'].fillna(answer_features['AvgConf'].median())
    # answer_features.fillna(-1, inplace=True)
    # answer_features = answer_features[answer_features.AvgConf > 0]
    solver_features = pd.read_csv(os.path.join(featured_dir_path, solver_features_file))
    solver_features['cad'] = solver_features['cad'].fillna(solver_features['cad'].median())
    solver_features['cadg'] = solver_features['cadg'].fillna(solver_features['cadg'].median())
    solver_features['Psolve'] = solver_features['Psolve'].fillna(solver_features['Psolve'].median())
    solver_features['arrogance'] = solver_features['arrogance'].fillna(solver_features['arrogance'].median())
    solver_features['Confidence'] = solver_features['Confidence'].fillna(solver_features['Confidence'].median())
    # solver_features.fillna(-1, inplace=True)
    # solver_features = solver_features[solver_features.Confidence > 0]
    return answer_features, solver_features


def all_data(featured_dir_path, answer_features_file, solver_features_file):
    print('{m} ALL_DATA : START {m}'.format(m='-'*100))

    answer_features, solver_features = get_normalized_features_df(featured_dir_path, answer_features_file, solver_features_file)

    dfs_list = []
    # answer_minFeatures_group_out_res, answer_minFeatures_group_out_preds = leave_one_out(answer_features, ANSWER_MIN_FEATURES)
    vals = {'aggregation': 'all_group', 'features': 'minFeatures', 'entity': 'answer'}
    # dfs_list.append(add_data_values(answer_minFeatures_group_out_res, vals))

    # solver_minFeatures_group_out_res, solver_minFeatures_group_out_preds = leave_one_out(solver_features, SOLVER_MIN_FEATURES)
    vals['entity'] = 'solver'
    # dfs_list.append(add_data_values(solver_minFeatures_group_out_res, vals))

    # ensemble_minFeatures_group_out_res = ensembles_leave_one_out(answer_features, ANSWER_MIN_FEATURES, solver_features, SOLVER_MIN_FEATURES)
    vals['entity'] = 'ensemble'
    # dfs_list.append(add_data_values(ensemble_minFeatures_group_out_res, vals))

    # answer_minFeatures_problem_out_res, answer_minFeatures_problem_out_preds = leave_one_out(answer_features, ANSWER_MIN_FEATURES, bool_problem=True)
    vals['aggregation'] = 'all_problem'
    vals['entity'] = 'answer'
    # dfs_list.append(add_data_values(answer_minFeatures_problem_out_res, vals))

    # solver_minFeatures_problem_out_res, solver_minFeatures_problem_out_preds = leave_one_out(solver_features, SOLVER_MIN_FEATURES, bool_problem=True)
    vals['entity'] = 'solver'
    # dfs_list.append(add_data_values(solver_minFeatures_problem_out_res, vals))

    # ensemble_minFeatures_problem_out_res = ensembles_leave_one_out(answer_features, ANSWER_MIN_FEATURES, solver_features, SOLVER_MIN_FEATURES, bool_problem=True)
    vals['entity'] = 'ensemble'
    # dfs_list.append(add_data_values(ensemble_minFeatures_problem_out_res, vals))

    # answer_features = answer_features[answer_features.AvgConf != -1]
    # solver_features = solver_features[solver_features.Confidence != 0]

    kwargs = {'answer': True}
    answer_Conf_group_out_res, answer_Conf_group_out_preds = leave_one_out(answer_features, ANSWER_FEATURES, **kwargs)
    vals['features'] = 'minPlus'
    vals['entity'] = 'answer'
    vals['aggregation'] = 'all_group'
    dfs_list.append(add_data_values(answer_Conf_group_out_res, vals))

    solver_Conf_group_out_res, solver_Conf_group_out_preds = leave_one_out(solver_features, SOLVER_FEATURES)
    vals['entity'] = 'solver'
    dfs_list.append(add_data_values(solver_Conf_group_out_res, vals))

    combined_Conf_group_out_res = combined_leave_one_out(answer_features, ANSWER_FEATURES, solver_features, SOLVER_FEATURES)
    vals['entity'] = 'ensemble'
    dfs_list.append(add_data_values(combined_Conf_group_out_res, vals))

    # answer_Conf_problem_out_res, answer_Conf_problem_out_preds = leave_one_out(answer_features, ANSWER_FEATURES, bool_problem=True)
    # vals['aggregation'] = 'all_problem'
    # vals['entity'] = 'answer'
    # dfs_list.append(add_data_values(answer_Conf_problem_out_res, vals))

    # solver_Conf_problem_out_res, solver_Conf_problem_out_preds = leave_one_out(solver_features, SOLVER_FEATURES, bool_problem=True)
    # vals['entity'] = 'solver'
    # dfs_list.append(add_data_values(solver_Conf_problem_out_res, vals))

    # ensemble_Conf_problem_out_res = combined_leave_one_out(answer_features, ANSWER_FEATURES, solver_features, SOLVER_FEATURES, bool_problem=True)
    # vals['entity'] = 'solver'
    # dfs_list.append(add_data_values(ensemble_Conf_problem_out_res, vals))

    try:
        for key in answer_Conf_group_out_preds.keys():
            # pd.concat(answer_minFeatures_group_out_preds[key]).to_csv(os.path.join(os.getcwd(), 'data', 'groups_problems', 'results', 'answer_minFeatures_group_{}.csv'.format(key)))
            # pd.concat(solver_minFeatures_group_out_preds[key]).to_csv(os.path.join(os.getcwd(), 'data', 'groups_problems', 'results', 'solver_minFeatures_group_{}.csv'.format(key)))
            # pd.concat(answer_minFeatures_problem_out_preds[key]).to_csv(os.path.join(os.getcwd(), 'data', 'groups_problems', 'results', 'answer_minFeatures_problem_{}.csv'.format(key)))
            # pd.concat(solver_minFeatures_problem_out_preds[key]).to_csv(os.path.join(os.getcwd(), 'data', 'groups_problems', 'results', 'solver_minFeatures_problem_{}.csv'.format(key)))
            pd.concat(answer_Conf_group_out_preds[key]).to_csv(os.path.join(os.getcwd(), 'data', 'groups_problems', 'results', 'answer_Conf_group_{}.csv'.format(key)))
            pd.concat(solver_Conf_group_out_preds[key]).to_csv(os.path.join(os.getcwd(), 'data', 'groups_problems', 'results', 'solver_Conf_group_{}.csv'.format(key)))
            # pd.concat(answer_Conf_problem_out_preds[key]).to_csv(os.path.join(os.getcwd(), 'data', 'groups_problems', 'results', 'answer_Conf_problem_{}.csv'.format(key)))
            # pd.concat(solver_Conf_problem_out_preds[key]).to_csv(os.path.join(os.getcwd(), 'data', 'groups_problems', 'results', 'solver_Conf_problem_{}.csv'.format(key)))
    except Exception as e:
        print(e)

    final_df = pd.concat(dfs_list)
    print('{m} ALL_DATA : OVER {m}'.format(m='-'*100))
    return final_df


def get_cluster(group, cluster_df):
    clustered = cluster_df[cluster_df.group_number == group]
    res = clustered.cluster.unique()
    zb = res[0] if len(res) else -2
    return zb


def clusterwise(featured_dir_path, answer_features_file, solver_features_file, cluster_dir_path):
    print('{m} CLUSTERS : START {m}'.format(m='-'*100))

    answer_features, solver_features = get_normalized_features_df(featured_dir_path, answer_features_file,
                                                                  solver_features_file)

    files = [os.path.join(cluster_dir_path, f) for f in os.listdir(cluster_dir_path) if f.endswith('.csv')]
    results = []
    for file_path in files:
        method = file_path.split(os.sep)[-1].strip('.csv').strip('cluster_')
        print('{t}{m} Clustering: {met} {m}'.format(t=' '*30, m='-'*70, met=method))

        cluster_df = pd.read_csv(file_path)
        # answer_features['cluster'] = answer_features.apply(
        #     lambda x: -2 if x['AvgConf'] == -1
        #     else cluster_df[cluster_df.group_number == x['group_number']].cluster.unique()[0], axis=1)
        # solver_features['cluster'] = solver_features.apply(
        #     lambda x: -2 if x['Confidence'] == 0
        #     else cluster_df[cluster_df.group_number == x['group_number']].cluster.unique()[0], axis=1)
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
            s = SOLVER_FEATURES
            solvv = solver_cluster_df[s]
            solvv2 = solvv.isnull().values
            solvv2 = solvv[solvv.isnull().values]
            answer_2_add, answer_pred_dict = leave_one_out(answer_cluster_df, ANSWER_FEATURES)
            solver_2_add, solver_pred_dict = leave_one_out(solver_cluster_df, SOLVER_FEATURES)
            ensemble_2_add = combined_leave_one_out(answer_cluster_df, ANSWER_FEATURES, solver_cluster_df, SOLVER_FEATURES)
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
            pd.concat(answer_pred_res[key]).to_csv(os.path.join(cluster_dir_path, 'results', '{method}_{model}_answer.csv'.format(method=method, model=key)))
            pd.concat(solver_pred_res[key]).to_csv(os.path.join(cluster_dir_path, 'results', '{method}_{model}_solver.csv'.format(method=method, model=key)))
    final_df = pd.concat(results)
    print('{m} CLUSTERS : OVER {m}'.format(m='-'*100))
    return final_df


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        features_dir = os.path.join(os.getcwd(), 'data', 'featured data')
        answer_feature_f = 'answer_features.csv'
        solver_feature_f = 'solver_features.csv'
        cluster_dir = os.path.join(os.getcwd(), 'data', 'clustered data')
        raw_data_dir = os.path.join(os.getcwd(), 'data', 'raw data')
        dataframes_list = [
            baseline(raw_data_dir),
            clusterwise(features_dir, answer_feature_f, solver_feature_f, cluster_dir),
            all_data(features_dir, answer_feature_f, solver_feature_f)
        ]
        pd.concat(dataframes_list).to_csv(os.path.join(os.getcwd(), 'data', 'no_arrogance_results.csv'))
