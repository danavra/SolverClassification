import pandas as pd
import numpy as np
import os
import warnings
from DataUtil import data_preparation, get_answers_from_df, NOT_ANSWERS, write_data

SOLVER_COLUMNS = ['Problem', 'Worker ID', 'Answer', 'group_number', 'psma', 'chisqr', 'bs', 'EAMA', 'EAAA', 'ec',
                  'arrogance', 'cad', 'cadg', 'Confidence', 'Subjective Difficulty', 'Psolve', 'Class']
ANSWER_COLUMNS = ['Answer', 'Problem', 'group_number', 'SD', 'AvgPSAR', 'AvgPSS', 'AvgPSNS', 'PSDAR', 'PSDS', 'PSDNS',
                  'AvgCS', 'AvgB', 'AvgPS', 'AvgConf', 'AvgDiff ', 'Class']


def calc_chisqr(x, ans_support_dict):
    """
    calculate the Chi square feature for a solver
    :param x: the solver entry
    :param ans_support_dict: (dict) the support distribution of the answers
    :return: (float) the Chi square calculation
    """
    ans = 0
    for key in ans_support_dict.keys():
        if (x[key] + ans_support_dict[key]) != 0:
            ans += ((x[key] - ans_support_dict[key]) ** 2) / (x[key] + ans_support_dict[key])
    return ans/2


def calc_bs(x, ans_support_dict):
    """
    calculate the Brier score feature for a solver
    :param x: the solver entry
    :param ans_support_dict: (dict) the support distribution of the answers
    :return:(float) the Brier score calculation
    """
    ans = 0
    for key in ans_support_dict.keys():
        ans += (x[key] - ans_support_dict[key]) ** 2
    return ans/len(ans_support_dict)


def calc_EAAA(x, ans_support_dict):
    """
    calculate the estimation ability all answers feature for a solver
    :param x: the solver entry
    :param ans_support_dict: (dict) the support distribution of the answers
    :return: (float) the EAAA calculation
    """
    ans = 0
    for key in ans_support_dict.keys():
        ans += abs(ans_support_dict[key] - x[key])
    return ans/len(ans_support_dict)


def nan(x):
    """
    returns nan
    :param x: doesn't matter
    :return: np.nan
    """
    return np.nan


def get_ans_class(ans, df):
    clas = df[df.Answer == ans]['Class'].unique()
    if clas and clas[0] == 1:
        return 1
    return 0


def extract_solver_features(df_raw, not_answers=NOT_ANSWERS, drop_cols=True):
    """
    extract the solver features and return pandas DataFrame
    :param df_raw: (pandas DataFrame) the raw data
    :param not_answers: (list) all the columns in the df which are not answers(str)
    :param drop_cols: (bool) True if want to drop unnecessary columns
    :return: (pandas DataFrame) contains all the featured data
    """
    global SOLVER_COLUMNS

    answers = get_answers_from_df(df_raw, not_answers)
    columns = df_raw.columns.values
    df = data_preparation(df_raw, not_answers)
    df['psma'] = df.apply(lambda x: x[str(x['Answer']).strip()], axis=1)
    group_size = len(df)
    ans_support_dict = dict([(ans, len(df[df.Answer == ans])/group_size) for ans in answers])
    df['chisqr'] = df.apply(lambda x: calc_chisqr(x, ans_support_dict), axis=1)
    df['bs'] = df.apply(lambda x: calc_bs(x, ans_support_dict), axis=1)
    df['EAMA'] = df.apply(lambda x: ans_support_dict[str(x['Answer']).strip()] - x[str(x['Answer']).strip()], axis=1)
    df['EAAA'] = df.apply(lambda x: calc_EAAA(x, ans_support_dict), axis=1)
    df['ec'] = df.apply(lambda x: x[str(x['Answer']).strip()] - (df[str(x['Answer']).strip()].sum() / len(df)), axis=1)

    if 'Confidence' in columns:
        # arr in [0, 0.9)
        df['arrogance'] = df.apply(
            lambda x: ((float(x['Confidence']) - 1) / 10) / (9 * (float(x[str(x['Answer']).strip()])) + 1), axis=1)
        # arr in [0, 1)
        # df['arrogance'] = df.apply(
        #     lambda x: ((float(x['Confidence']) - 1) / 10) / (9 * (float(x[str(x['Answer']).strip()])) + 0.9), axis=1)
        df['Confidence'] = df['Confidence'].apply(lambda x: x / 10)
        df['cad'] = df['Confidence'].apply(lambda x: x - df['Confidence'].mean())
        df['cadg'] = df.apply(lambda x: x['Confidence'] - df[df['Answer'] == x['Answer']]['Confidence'].mean(), axis=1)
    else:
        df['arrogance'] = df.apply(nan, axis=1)
        df['Confidence'] = df.apply(nan, axis=1)
        df['cad'] = df.apply(nan, axis=1)
        df['cadg'] = df.apply(nan, axis=1)

    if 'Subjective Difficulty' not in columns:
        df['Subjective Difficulty'] = df.apply(nan, axis=1)

    if drop_cols:
        cols_to_drop = [col for col in df.columns.values if (col not in SOLVER_COLUMNS) and (col not in answers)]
        df.drop(cols_to_drop, axis=1, inplace=True)

    return df


def extract_answer_features(df_raw, not_answers=NOT_ANSWERS, include_solver=False):
    """
    extract the answer features and return pandas DataFrame
    :param df_raw: (pandas DataFrame) the raw data
    :param not_answers: (list) all the columns in the df which are not answers(str)
    :param include_solver: (bool) does the df_raw includes the solver features
    :return: (pandas DataFrame) contains all the featured data
    """
    global ANSWER_COLUMNS
    answers = get_answers_from_df(df_raw, not_answers)
    columns = df_raw.columns.values
    answers_count = len(answers)
    df = df_raw if include_solver else extract_solver_features(df_raw, not_answers, drop_cols=False)
    df['Answer'] = df['Answer'].apply(str)
    new_df = pd.DataFrame(answers, columns=['Answer'])
    new_df['Problem'] = df['Problem'].array[:answers_count]
    new_df['group_number'] = df['group_number'].array[:answers_count]
    new_df['Class'] = new_df['Answer'].apply(lambda x: get_ans_class(x, df))
    new_df['SD'] = new_df['Answer'].apply(lambda x: len(df[df.Answer == x]) / len(df))
    new_df['AvgPSAR'] = new_df['Answer'].apply(lambda x: df[x].mean())
    new_df['AvgPSS'] = new_df['Answer'].apply(lambda x: df[df.Answer == x][x].mean() if len(df[df.Answer == x]) else 0)
    new_df['AvgPSNS'] = new_df['Answer'].apply(lambda x: df[df.Answer != x][x].mean())
    new_df['PSDAR'] = new_df.apply(lambda x: x.SD - x.AvgPSAR, axis=1)
    new_df['PSDS'] = new_df.apply(lambda x: x.SD - x.AvgPSS, axis=1)
    new_df['PSDNS'] = new_df.apply(lambda x: x.SD - x.AvgPSNS, axis=1)
    new_df['AvgCS'] = new_df['Answer'].apply(
        lambda x: df[df.Answer == x]['chisqr'].mean() if len(df[df.Answer == x]) else 0)
    new_df['AvgB'] = new_df['Answer'].apply(lambda x: df[df.Answer == x]['bs'].mean() if len(df[df.Answer == x]) else 0)

    conf_nan = any(map(np.isnan, df['Confidence'].unique()))
    extract_conf = (include_solver and not conf_nan) or (not include_solver and 'Confidence' in columns)
    if extract_conf:
        new_df['AvgPS'] = new_df['Answer'].apply(
            lambda x: df[df.Answer == x]['Psolve'].mean() if len(df[df.Answer == x]) else 0)
        new_df['AvgConf'] = new_df['Answer'].apply(
            lambda x: df[df.Answer == x]['Confidence'].mean() if len(df[df.Answer == x]) else 0)
    else:
        new_df['AvgPS'] = new_df.apply(nan, axis=1)
        new_df['AvgConf'] = new_df.apply(nan, axis=1)

    diff_nan = any(map(np.isnan, df['Subjective Difficulty'].unique()))
    extract_diff = (include_solver and not diff_nan) or (not include_solver and 'Subjective Difficulty' in columns)
    if extract_diff:
        new_df['AvgDiff'] = new_df['Answer'].apply(
            lambda x: df[df.Answer == x]['Subjective Difficulty'].mean() if len(df[df.Answer == x]) else 0)
    else:
        new_df['AvgDiff'] = new_df.apply(nan, axis=1)

    return new_df.set_index('Answer')


def extract_features_from_dir(dir_path, not_answers=NOT_ANSWERS):
    """
    extracting solver and answer features for all the csv files in a given directory path and write the featured data
     to the given directory with the names 'solver_features' and 'answer'features'
    :param dir_path: the path of the directory
    :param not_answers: (list(str)) list of columns' names that are not the answers of the problem(given default)
    :return: tuple(pandas.DataFrame, pandas,DataFram) the solver, answer featured data
    """
    files_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if
                   os.path.isfile(os.path.join(dir_path, file)) and file.endswith('.csv')]

    if not files_paths:
        return None, None

    list_of_dfs_solver = []
    list_of_dfs_answer = []

    for file_path in files_paths:
        df_raw = pd.read_csv(file_path, dtype={'Answer': str})
        groups = df_raw.group_number.unique()
        for group in groups:
            group_df = df_raw[df_raw.group_number == group]
            answers = get_answers_from_df(group_df, not_answers)
            solver_df = extract_solver_features(group_df, not_answers)
            not_answers += SOLVER_COLUMNS
            answer_df = extract_answer_features(solver_df, not_answers, include_solver=True)
            cols_to_drop = [col for col in solver_df.columns.values if (col not in SOLVER_COLUMNS) and (col not in answers)]
            solver_df.drop(cols_to_drop, axis=1, inplace=True)
            list_of_dfs_solver.append(solver_df)
            list_of_dfs_answer.append(answer_df)

    all_data_solver = pd.concat(list_of_dfs_solver)
    all_data_ans = pd.concat(list_of_dfs_answer)
    return all_data_solver, all_data_ans


def feature_extraction():
    raw_data_path = os.path.join(os.getcwd(), 'data', 'raw data')
    solver_features, answer_features = extract_features_from_dir(raw_data_path)
    if solver_features is not None and answer_features is not None:
        write_data(solver_features, os.path.join(os.getcwd(), 'data', 'featured data', 'solver_features.csv'))
        write_data(answer_features, os.path.join(os.getcwd(), 'data', 'featured data', 'answer_features.csv'))


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # feature_extraction()
