from os import listdir, getcwd
from os.path import isfile, join
import pandas as pd
import warnings


def getAllFilesInFolder(folderPath):
    onlyCsvs = [folderPath+'\\'+ f for f in listdir(folderPath) if isfile(join(folderPath, f)) and f.split('.')[1]=='csv']
    return onlyCsvs


def mergeLastDfs(list_of_dfs):
    ndf = pd.concat([list_of_dfs[len(list_of_dfs) - 2], list_of_dfs[len(list_of_dfs) - 1]])
    del list_of_dfs[len(list_of_dfs) - 1]
    del list_of_dfs[len(list_of_dfs) - 1]
    list_of_dfs.append(ndf)
    return list_of_dfs


def seperateToGroups(csv_file,groupNum):
    df=pd.read_csv(csv_file)
    numOfGroups=int(len(df)/30)
    numOfAdditionalSubjects=len(df)%30
    list_of_dfs=[]
    if numOfAdditionalSubjects<10:
        list_of_dfs = [df.loc[i:i+30-1,:] for i in range(0, len(df),30)]
        list_of_dfs=mergeLastDfs(list_of_dfs)
    elif numOfGroups<=numOfAdditionalSubjects:
        num_of_subjects_per_group=30+int(numOfAdditionalSubjects/numOfGroups)
        list_of_dfs = [df.loc[i:i + num_of_subjects_per_group - 1, :] for i in range(0, len(df), num_of_subjects_per_group)]
        if(len(list_of_dfs[len(list_of_dfs) - 1])<30):
            list_of_dfs=mergeLastDfs(list_of_dfs)
    else:
        list_of_dfs = [df.loc[i:i + 30 - 1, :] for i in range(0, len(df), 30)]
        last_df=list_of_dfs[len(list_of_dfs) - 1]
        del list_of_dfs[len(list_of_dfs) - 1]
        last_df_separated=[last_df[i:i+3] for i in range(0, len(last_df), 3)]
        for small_df in last_df_separated:
            ndf=pd.concat([list_of_dfs[0], small_df])
            del list_of_dfs[0]
            list_of_dfs.append(ndf)
    for i in range(0,len(list_of_dfs)):
        currDf=list_of_dfs[i]
        currDf['group_number']=groupNum
        list_of_dfs[i]=currDf
        groupNum+=1
    data = pd.concat(list_of_dfs)
    data.set_index('Worker ID', inplace=True)
    data.to_csv(f)
    return groupNum
def makeGroups():
    files=getAllFilesInFolder('C:\\Users\\Administrator\\PycharmProjects\\final project')
    groupNum=1
    for f in files:
        groupNum=seperateToGroups(f,groupNum)

########################################feature extraction ###################################################

not_answers = ['Gender','Education','Confidence','Problem','Worker ID','Age','Strong hand','Psolve','Psolve','Subjective Difficulty','Objective Difficutly','Class','group_number','Answer']
final_features = ['Problem', 'Worker ID', 'Answer', 'group_number', 'psma', 'chisqr', 'bs', 'EAMA', 'EAAA', 'ec',
                      'arrogance', 'cad', 'cadg', 'Confidence', 'Subjective Difficulty', 'Psolve', 'Class']
# final_features = ['Problem', 'Worker ID', 'Answer', 'group_number', 'psma', 'chisqr', 'bs', 'EAMA', 'EAAA', 'ec',
#                       'arrogance', 'cad', 'cadg', 'Confidence', 'Psolve', 'Class']
data_path = join(getcwd(), 'data', 'raw data')
files = getAllFilesInFolder(data_path)
precentageDict = {
    "0-10%": 0.1,
    "11-20%": 0.2,
    "21-30%": 0.3,
    "31-40%": 0.4,
    "41-50%": 0.5,
    "51-60%": 0.6,
    "61-70%": 0.7,
    "71-80%": 0.8,
    "81-90%": 0.9,
    "91-100%": 1.0,
    "51-100%": 0.75
}


def change_precentage_to_num(df, answers):
    for ans in answers:
        if isinstance(df.iloc[1][ans], str) and ('-' in df.iloc[1][ans]):
            df[ans] = df[ans].apply(lambda x: precentageDict[x])
        elif isinstance(df.iloc[1][ans], str) and ('%' in df.iloc[1][ans]):
            df[ans] = df[ans].apply(lambda x: int(x.strip('%'))/100)
    if 'Psolve' in df.columns.values:
        if isinstance(df.iloc[1]['Psolve'], str) and ('-' in df.iloc[1]['Psolve']):
            df['Psolve'] = df['Psolve'].apply(lambda x: precentageDict[x])
        elif isinstance(df.iloc[1]['Psolve'], str) and ('%' in df.iloc[1]['Psolve']):
            df['Psolve'] = df['Psolve'].apply(lambda x: int(x.strip('%')) / 100)
    return df


def psma(df):
    df['psma'] = df.apply(lambda x: x[str(x['Answer']).strip()], axis=1)
    return df


def calc_chisqr(x, dic):
    ans = 0
    for key in dic.keys():
        if (x[key] + dic[key]) != 0:
            ans += ((x[key]-dic[key])**2)/(x[key]+dic[key])
    return ans/2


def chi_sqr(df, dic):
    df['chisqr'] = df.apply(lambda x: calc_chisqr(x, dic), axis=1)
    return df


def calc_bs(x, dic):
    ans=0
    for key in dic.keys():
        ans += (x[key]-dic[key])**2
    return ans/len(dic)


def brier(df, dic):
    df['bs'] = df.apply(lambda x: calc_bs(x, dic), axis=1)
    return df


def getEAMA(df, dic):
    df['EAMA'] = df.apply(lambda x: dic[str(x['Answer']).strip()]-x[str(x['Answer']).strip()], axis=1)
    return df


def calc_EAAA(x, dic):
    ans = 0
    for key in dic.keys():
        ans += abs(dic[key]-x[key])
    return ans/len(dic)


def getEAAA(df, dic):
    df['EAAA'] = df.apply(lambda x: calc_EAAA(x,dic), axis=1)
    return df


def getEC(df):
    df['ec'] = df.apply(lambda x: x[str(x['Answer']).strip()]-(df[str(x['Answer']).strip()].sum()/len(df)), axis=1)
    return df


def get_arrogance(df):
    df['arrogance'] = df.apply(lambda x:
                               ((float(x['Confidence']) - 1) / 10) / (9 * (float(x[str(x['Answer']).strip()])) + 1),
                               axis=1)
    return df


def get_cads(df):
    df['cad'] = df['Confidence'].apply(lambda x: x-df['Confidence'].mean())
    df['cadg'] = df.apply(lambda x: x['Confidence']-df[df['Answer'] == x['Answer']]['Confidence'].mean(), axis=1)
    return df


def get_solver_features(answers, df):
    global final_features
    df = change_precentage_to_num(df, answers)
    df = psma(df)
    dic = dict()
    group_size = len(df)
    for ans in answers:
        dic[ans] = len(df[df['Answer'] == ans]) / group_size
    df = chi_sqr(df, dic)
    df = brier(df, dic)
    df = getEAMA(df, dic)
    df = getEAAA(df, dic)
    df = getEC(df)
    df = get_arrogance(df)
    df['Confidence'] = df['Confidence'].apply(lambda x: x / 10)
    if 'Subjective Difficulty' in df.columns.values:
        df['Subjective Difficulty'] = df['Subjective Difficulty'].apply(lambda x: x / 10)
    df = get_cads(df)
    df['Class'] = df['Class'].apply(lambda x: 0 if 'no' in x.lower() else 1)

    cols_to_drop = [col for col in df.columns.values if (col not in final_features) and (col not in answers)]
    return df.drop(cols_to_drop, axis=1)


def get_ans_class(ans, df):
    clas = df[df.Answer == ans]['Class'].unique()
    if clas and clas[0] == 1:
        return 1
    return 0


def get_answer_features(answers, df):
    ans_count = len(answers)
    df['Answer'] = df['Answer'].apply(str)
    new_df = pd.DataFrame(answers, columns=['Answer'])
    new_df['Problem'] = df['Problem'].array[:ans_count]
    new_df['group_number'] = df['group_number'].array[:ans_count]
    new_df['Class'] = new_df['Answer'].apply(lambda x: get_ans_class(x, df))
    new_df['SD'] = new_df['Answer'].apply(lambda x: len(df[df.Answer == x])/len(df))
    new_df['AvgPSAR'] = new_df['Answer'].apply(lambda x: df[x].mean())
    new_df['AvgPSS'] = new_df['Answer'].apply(lambda x: df[df.Answer == x][x].mean() if len(df[df.Answer == x]) else 0)
    new_df['AvgPSNS'] = new_df['Answer'].apply(lambda x: df[df.Answer != x][x].mean())
    new_df['PSDAR'] = new_df.apply(lambda x: x.SD - x.AvgPSAR, axis=1)
    new_df['PSDS'] = new_df.apply(lambda x: x.SD - x.AvgPSS, axis=1)
    new_df['PSDNS'] = new_df.apply(lambda x: x.SD - x.AvgPSNS, axis=1)
    new_df['AvgCS'] = new_df['Answer'].apply(
        lambda x: df[df.Answer == x]['chisqr'].mean() if len(df[df.Answer == x]) else 0)
    new_df['AvgB'] = new_df['Answer'].apply(lambda x: df[df.Answer == x]['bs'].mean() if len(df[df.Answer == x]) else 0)
    new_df['AvgPS'] = new_df['Answer'].apply(
        lambda x: df[df.Answer == x]['Psolve'].mean() if len(df[df.Answer == x]) else 0)
    new_df['AvgConf'] = new_df['Answer'].apply(
        lambda x: df[df.Answer == x]['Confidence'].mean() if len(df[df.Answer == x]) else 0)
    # if 'Subjective Difficulty' in df.columns.values:
    #     new_df['AvgDiff'] = new_df['Answer'].apply(
    #         lambda x: df[df.Answer == x]['Subjective Difficulty'].mean() if len(df[df.Answer == x]) else 0)
    return new_df.set_index('Answer')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # list_of_dfs_solver = []
        list_of_dfs_answer = []
        for f in files:
            df = pd.read_csv(f, index_col='Worker ID')
            cols = list(df.columns.values)
            answers = [col for col in cols if col not in not_answers]
            groups = df.group_number.unique()
            for group in groups:
                df_2_add = get_solver_features(answers, df[df['group_number'] == group])
                list_of_dfs_answer.append(get_answer_features(answers, df_2_add))
                # list_of_dfs_solver.append(df_2_add.drop(answers, axis=1))

        # all_data_solver = pd.concat(list_of_dfs_solver)
        all_data_ans = pd.concat(list_of_dfs_answer)
        # all_data_solver.to_csv('subjects_features_allFeatures.csv')
        all_data_ans.to_csv('answers_features_Conf.csv')

        print('*********************************************************')
