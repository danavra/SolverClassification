import pandas as pd
import math
import statistics
import json
from os import getcwd, listdir, sep
from os.path import join
import warnings
import final_project as fp


def get_highest_voted_ans(dic_prob):
    max_votes = 0
    highest_voted_ans = ''
    for ans in dic_prob:
        if dic_prob[ans] > max_votes:
            max_votes = dic_prob[ans]
            highest_voted_ans = ans
    return dic_prob[highest_voted_ans]


def get_consensus(dic_prob):
    consensus = 0
    for p_name, p in dic_prob.items():
        if p == 0:
            continue
        consensus -= p * math.log2(p)
    return 1 - (consensus / math.log2(len(dic_prob)))


def get_answer_distribution(all_possible_ans, df):
    dic = {}
    # initialize with 0
    for ans in all_possible_ans:
        dic[str(ans)] = 0

    # count answers
    for answer in df['Answer']:
        dic[str(answer)] += 1
    return dic


def get_agg_arrogance(df):
    temp = fp.get_arrogance(df)
    all_arrogance = temp['arrogance']
    return statistics.mean(all_arrogance), statistics.median(all_arrogance), statistics.variance(all_arrogance)


def get_agg_confidence(df):
    confidence = df['Confidence'].apply(lambda x: x / 10)
    return statistics.mean(confidence), statistics.median(confidence), statistics.variance(confidence)


def get_agg_EMAM(df, dic):
    temp = fp.getEAMA(df,dic)['EAMA']
    normalized_eama = temp.apply((lambda x: (x+1)/2))
    return statistics.mean(normalized_eama), statistics.median(normalized_eama), statistics.variance(normalized_eama)


def get_agg_EAAA(df, dic):
    temp = fp.getEAAA(df,dic)['EAAA']
    return statistics.mean(temp), statistics.median(temp), statistics.variance(temp)


def get_init_features(df, num_of_answers, first_idx):
    features = {}
    all_possible_ans = df.columns[first_idx:first_idx + num_of_answers]
    df = fp.change_precentage_to_num(df, all_possible_ans)
    dic = get_answer_distribution(all_possible_ans, df)

    # get probability
    num_of_subjects = len(df['Answer'])
    dic_prob = {}
    for ans in dic:
        dic_prob[ans] = dic[ans]/num_of_subjects

    # simple meta features extraction
    features['consensus'] = get_consensus(dic_prob) # divided by log(n) 0: full consensus 1: no consensus
    features['highest_voted_ans'] = get_highest_voted_ans(dic_prob) # the percentage (amount / all) of the highest voted answer
    features['variance'] = statistics.variance(list(dic.values())) # variance of the problems answers

    # complex meta features extraction (for each feature: avg, median and var (normalized by max var))
    features['avg_arrogance'], features['med_arrogance'], features['var_arrogance'] = get_agg_arrogance(df)
    features['avg_confidence'], features['med_confidence'], features['var_confidence'] = get_agg_confidence(df)
    features['avg_EMAM'], features['med_EMAM'], features['var_EMAM'] = get_agg_EMAM(df,dic_prob)
    features['avg_EAAA'], features['med_EAAA'], features['var_EAAA'] = get_agg_EAAA(df,dic_prob)

    return features


def normalize_var(df, col_names):
    """
    Given a data frame and a list of column name, will normalize all the data under each given
    column name between 0 - 1
    :param df: the data frame to apply changes to
    :param col_names: the name of the columns in the data frame that are to be normalized
    :return: the data frame normalized
    """
    for feature in col_names:
        max = df[feature].max()
        min = df[feature].min()
        df[feature] = df[feature].apply(lambda x: (x - min)/(max - min))
    return df


def extract_meta_features(all_prob_dir, json_meta_data):
    """
    Calculates the meta features of all the problems in the gived directory.
    :param all_prob_dir: the path of the directory that hold all of the problems data sets
    :param json_meta_data: a json file that holds for each problem the answers starting index
           and the answer amount
    :return: a data frame with all the problems meta features (normalized between 0,1)
    """
    with open(json_meta_data) as json_file:
        problem_dic = json.load(json_file)
        ans = {}
        all_prob_dir_path = join(getcwd(), 'data', 'raw data')

        # # collect all problems that have meta data.
        valid_problems = []
        for prob in problem_dic:
            valid_problems.append(prob)

        # iterate over every problem and extract meta features
        for problem_path in all_prob_dir:
            file_name = join(all_prob_dir_path, problem_path)
            prob_df = pd.read_csv(file_name, index_col='Worker ID', dtype={'Answer': str})
            for group in prob_df.group_number.unique():
                df = prob_df[prob_df.group_number == group]
                problem_name = df['Problem'].unique()[0]
                if problem_name not in problem_dic:
                    continue

                # collect needed meta data from JSON
                num_of_answers = problem_dic[problem_name]['num_of_answers']
                first_idx = problem_dic[problem_name]['first_idx']

                # get initial features
                ans['{g}'.format(g=group)] = get_init_features(df, num_of_answers, first_idx)

        # change  dic to df
        ans_df = pd.DataFrame(ans).transpose()

        # normalize variance features [0,1]
        to_normalize = ['variance', 'var_EAAA', 'var_EMAM', 'var_confidence', 'var_arrogance']
        ans_df = normalize_var(ans_df, to_normalize)
        return ans_df


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # default values
        json_meta_data = join(getcwd(), 'data', 'meta data', 'problems_meta_data.JSON')
        all_prob_dir = listdir(join(getcwd(), 'data', 'raw data'))

        # get meta features
        m_features = extract_meta_features(all_prob_dir, json_meta_data)
        print("Success:\n" + str(m_features))

        # add index name (group_number)
        m_features.index.names = ['group_number']
        # export_features_csv(m_features)
        m_features.to_csv(join(getcwd(), 'data', 'meta data', 'meta_features.csv'))

