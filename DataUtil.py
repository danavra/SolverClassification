import pandas as pd
import os
import warnings

PERCENTAGE_DICT = {"0-10%": 0.1, "11-20%": 0.2, "21-30%": 0.3, "31-40%": 0.4, "41-50%": 0.5, "51-60%": 0.6,
                   "61-70%": 0.7, "71-80%": 0.8, "81-90%": 0.9, "91-100%": 1.0, "51-100%": 0.75}
NOT_ANSWERS = ['Gender', 'Education', 'Confidence', 'Problem', 'Worker ID', 'Age', 'Strong hand', 'Psolve', 'Hand',
               'Subjective Difficulty', 'Objective Difficutly', 'Class', 'group_number', 'Answer']


def get_answers_from_df(df_raw, not_answers=NOT_ANSWERS):
    """
    get the answers from the df
    :param df_raw: (pandas DataFrame) the raw data
    :param not_answers: (list) all the columns in the df which are not answers(str)
    :return: (list) all the answers(str)
    """
    cols = list(df_raw.columns.values)
    answers = [col for col in cols if col not in not_answers]
    return answers


def data_preparation(df_raw, not_answers=NOT_ANSWERS):
    """
    prepare the data
    :param df_raw: (pandas DataFrame) the raw data
    :param not_answers: (list) all the columns in the df which are not answers(str)
    :return: (pandas DataFrame) the raw data with prepared data(normalized and in the right type)
    """
    global PERCENTAGE_DICT
    answers = get_answers_from_df(df_raw, not_answers)
    for ans in answers:
        if isinstance(df_raw.iloc[1][ans], str) and ('-' in df_raw.iloc[1][ans]):
            df_raw[ans] = df_raw[ans].apply(lambda x: PERCENTAGE_DICT[x])
        elif isinstance(df_raw.iloc[1][ans], str) and ('%' in df_raw.iloc[1][ans]):
            df_raw[ans] = df_raw[ans].apply(lambda x: int(x.strip('%'))/100)

    columns = df_raw.columns.values
    if 'Psolve' in columns:
        if isinstance(df_raw.iloc[1]['Psolve'], str) and ('-' in df_raw.iloc[1]['Psolve']):
            df_raw['Psolve'] = df_raw['Psolve'].apply(lambda x: PERCENTAGE_DICT[x])
        elif isinstance(df_raw.iloc[1]['Psolve'], str) and ('%' in df_raw.iloc[1]['Psolve']):
            df_raw['Psolve'] = df_raw['Psolve'].apply(lambda x: int(x.strip('%')) / 100)

    if 'Subjective Difficulty' in columns:
        df_raw['Subjective Difficulty'] = df_raw['Subjective Difficulty'].apply(lambda x: x / 10)

    df_raw['Class'] = df_raw['Class'].apply(lambda x: 0 if 'no' in x.lower() else 1)
    return df_raw


def merge_last_dfs(list_of_dfs):
    """
    merging the dfs back
    :param list_of_dfs: (list(pandas DataFrame)) list of all the dfs to merge
    :return: list of dfs with the last 2 merged
    """
    ndf = pd.concat([list_of_dfs[len(list_of_dfs) - 2], list_of_dfs[len(list_of_dfs) - 1]])
    del list_of_dfs[len(list_of_dfs) - 1]
    del list_of_dfs[len(list_of_dfs) - 1]
    list_of_dfs.append(ndf)
    return list_of_dfs


def check_solvers_in_df(list_of_dfs):
    new_df_list = []
    for df in list_of_dfs:
        if len(df[df['Class'] == 'Solver']) == 0:
            for i, ndf in enumerate(list_of_dfs):
                if len(ndf[ndf['Class'] == 'Solver']) > 1:
                    solver_id=ndf[ndf['Class'] == 'Solver']['Worker ID'].unique()[0]
                    solver_df=ndf[ndf['Worker ID'] == solver_id]
                    ndf=ndf[ndf['Worker ID'] != solver_id]
                    del list_of_dfs[i]
                    df=pd.concat([df, solver_df])
                    new_df_list.append(df)
                    new_df_list.append(ndf)
        else:
            new_df_list.append(df)
    return new_df_list


def separate_to_groups(df_path, group_num):
    """
    seperate a df to groups by given group number
    :param df_path: the path to the df
    :param group_num: the initial group number
    :return: the next group number
    """
    df = pd.read_csv(df_path, dtype={'Answer': str})
    numOfGroups = int(len(df) / 30)
    numOfAdditionalSubjects = len(df) % 30
    list_of_dfs = []
    if numOfAdditionalSubjects < 10:
        list_of_dfs = [df.loc[i:i+29, :] for i in range(0, len(df), 30)]
        list_of_dfs = merge_last_dfs(list_of_dfs)
    elif numOfGroups <= numOfAdditionalSubjects:
        num_of_subjects_per_group = 30 + int(numOfAdditionalSubjects / numOfGroups)
        list_of_dfs = [df.loc[i:i+num_of_subjects_per_group - 1, :] for i in
                       range(0, len(df), num_of_subjects_per_group)]
        if len(list_of_dfs[len(list_of_dfs) - 1]) < 30:
            list_of_dfs = merge_last_dfs(list_of_dfs)
    else:
        list_of_dfs = [df.loc[i:i+29, :] for i in range(0, len(df), 30)]
        last_df = list_of_dfs[len(list_of_dfs) - 1]
        del list_of_dfs[len(list_of_dfs) - 1]
        last_df_separated = [last_df[i:i + 3] for i in range(0, len(last_df), 3)]
        for small_df in last_df_separated:
            ndf = pd.concat([list_of_dfs[0], small_df])
            del list_of_dfs[0]
            list_of_dfs.append(ndf)

    list_of_dfs = check_solvers_in_df(list_of_dfs)
    for i in range(0, len(list_of_dfs)):
        currDf = list_of_dfs[i]
        currDf['group_number'] = group_num
        list_of_dfs[i] = currDf
        group_num += 1

    data = pd.concat(list_of_dfs)
    data.set_index('Worker ID', inplace=True)
    data.to_csv(df_path)
    return group_num


def make_groups(dir_path):
    """
    given a directory path divide all the dfs to group (add to the file another column to 'group_number')
    :param dir_path: the path to the directory
    """
    files_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if
                   os.path.isfile(os.path.join(dir_path, file)) and file.endswith('.csv')]
    group_num = 1
    for file_path in files_paths:
        group_num = separate_to_groups(file_path, group_num)
        print(file_path)


def write_data(df, output_path):
    """
    writing the given DataFrame to a csv file
    :param df: (pandas DataFrame) the given data
    :param output_path: the path to write to
    """
    df.to_csv(output_path)


def directories_validation():
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    clust = os.path.join(data_dir, 'clustered data')
    if not os.path.isdir(clust):
        os.mkdir(clust)
        os.mkdir(os.path.join(clust, 'results'))
    elif not os.path.isdir(os.path.join(clust, 'results')):
        os.mkdir(os.path.join(clust, 'results'))

    feat = os.path.join(data_dir, 'featured data')
    if not os.path.isdir(feat):
        os.mkdir(feat)

    meta = os.path.join(data_dir, 'meta data')
    if not os.path.isdir(meta):
        os.mkdir(meta)

    preds = os.path.join(data_dir, 'predictions')
    if not os.path.isdir(preds):
        os.mkdir(preds)

    raw = os.path.join(data_dir, 'raw data')
    if not os.path.isdir(raw):
        os.mkdir(raw)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # raw_data_dir_path = os.path.join(os.getcwd(), 'data', 'raw data')
        # make_groups(raw_data_dir_path)
        # mark = '*'*60
        # print('{0} DONE {0}'.format(mark))
        print('start')
        directories_validation()
        print('end')
