import pandas as pd
import numpy as np
import os
import warnings
from final_project import not_answers, change_precentage_to_num


RAW_DATA_PARH = os.path.join(os.getcwd(), 'data', 'raw data')


def majority_rule(df, possible_answers=None):
    return df['Answer'].value_counts().index[0]


def surprisingly_popular(df: pd.DataFrame, possible_answers):
    # for i in df.iterrows():
    #     j = i[1][possible_answers]
    #     print(i[1][possible_answers].idmax())
    df['sp_ans'] = df[possible_answers].idxmax(axis=1)
    return df['sp_ans'].value_counts().index[0]


def average_confidence(df: pd.DataFrame, possible_answers):
    answers_conf = dict()
    for ans in possible_answers:
        answers_conf[ans] = df[df.Answer == ans]['Confidence'].mean()

    return max(answers_conf, key=answers_conf.get)


def confidence_weighted(df: pd.DataFrame, possible_answers):
    answers_conf = dict()
    for ans in possible_answers:
        answers_conf[ans] = df[df.Answer == ans]['Confidence'].sum()

    return max(answers_conf, key=answers_conf.get)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data_files = [file for file in os.listdir(RAW_DATA_PARH)]
        success = {'mr': 0, 'sp': 0, 'avg_conf': 0, 'w_conf': 0}
        groups = 0
        for data_file in data_files:
            df = pd.read_csv(os.path.join(RAW_DATA_PARH, data_file))
            cols = list(df.columns.values)
            answers = [col for col in cols if col not in not_answers]
            df = change_precentage_to_num(df, answers)
            for group in df.group_number.unique():
                groups += 1
                group_df = df[df.group_number == group]
                right_ans = group_df[group_df.Class == 'Solver']['Answer'].values[0]
                mr = majority_rule(group_df, answers)
                sp = surprisingly_popular(group_df, answers)
                avg_conf = average_confidence(group_df, answers)
                w_conf = confidence_weighted(group_df, answers)
                success['mr'] = success['mr']+1 if mr == right_ans else success['mr']
                success['sp'] = success['sp']+1 if sp == right_ans else success['sp']
                success['avg_conf'] = success['avg_conf']+1 if avg_conf == right_ans else success['avg_conf']
                success['w_conf'] = success['w_conf']+1 if avg_conf == right_ans else success['w_conf']
        print(success)
        print('num of groups: {}'.format(groups))