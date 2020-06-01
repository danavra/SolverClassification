import pandas as pd
from sklearn.base import clone as clone_model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import os
import warnings
from ModelTesting import get_models
from ModelTesting import ANSWER_FEATURES, SOLVER_FEATURES

ANSWER_MIN_FEATURES = ANSWER_FEATURES[:-2]
SOLVER_MIN_FEATURES = SOLVER_FEATURES[:-5]


class CombinedModel:
    """
    class Ensemble model with answer's and solver's features - generic
    """
    def __init__(self, answer_dict, solver_dict):
        """
        initializing model
        :param answer_dict: (dict) this form: {'model1': Model, 'model2': Model, 'weights': [w1, w2]}
        :param solver_dict: (dict) this form: {'model1': Model, 'model2': Model, 'weights': [w1, w2]}
        """
        self.models = dict()
        self.weights = dict()
        for key in answer_dict.keys():
            if key != 'weights':
                num = key.strip('model')
                self.models['answer{}'.format(num)] = answer_dict[key]
                self.weights['answer{}'.format(num)] = answer_dict['weights'][int(num)-1]

        for key in solver_dict.keys():
            if key != 'weights':
                num = key.strip('model')
                self.models['solver{}'.format(num)] = solver_dict[key]
                self.weights['solver{}'.format(num)] = solver_dict['weights'][int(num)-1]

    def fit_models(self, X_answer, y_answer, X_solver, y_solver):
        for key in self.models.keys():
            if 'answer' in key:
                self.models[key].fit(X_answer, y_answer)
            else:
                self.models[key].fit(X_solver, y_solver)

    def get_model_solution(self, model, X):
        X = pd.DataFrame.copy(X)
        X_test = X.drop(['Answer'], axis=1)
        X['pred'] = model.predict(X_test)
        model_ans = None
        if len(X[X.pred == True]) != 0:
            ans_count = X[X.pred == True]['Answer'].value_counts()
            if len(ans_count.index) == 1 or (len(ans_count.index) > 1 and ans_count.values[0] > ans_count.values[1]):
                model_ans = ans_count.index[0]

        if model_ans is None:
            probas = model.predict_proba(X_test)
            probas = list(map(lambda x: x[1], probas))
            X['probas'] = probas
            model_ans = X[X.probas == X.probas.max()]['Answer'].values[0]

        return model_ans

    def find_solution(self, X_answer, X_solver):
        solutions = dict()
        final_solution = dict()

        for key in self.models.keys():
            if 'answer' in key:
                solutions[key] = self.get_model_solution(self.models[key], X_answer)
            else:
                solutions[key] = self.get_model_solution(self.models[key], X_solver)

        total_weights = sum(self.weights.values())

        for model, solution in solutions.items():
            if solution in final_solution.keys():
                final_solution[solution] += self.weights[model]/total_weights
            else:
                final_solution[solution] = self.weights[model]/total_weights

        return max(final_solution, key=final_solution.get)


class CombinedAll(CombinedModel):
    """
    class Ensemble model with answer's and solver's features for all data
    """
    def __init__(self):
        pass


class CombinedCluster(CombinedModel):
    """
    class Ensemble model with answer's and solver's features by cluster
    """
    def __init__(self):
        pass


def combined_get_models():
    kwargs = {'answer': True}
    models = get_models(**kwargs)
    ensemble_models = dict()
    answer_dict = {'model1': LinearDiscriminantAnalysis(), 'model2': clone_model(models['Vote_soft']), 'weights': [0.2, 0.2]}
    models = get_models()
    solver_dict = {'model1': clone_model(models['Vote_soft']), 'model2': clone_model(models['Bag_KNN']), 'weights': [0.4, 0.2]}
    ensemble_models['Combined'] = CombinedModel(answer_dict, solver_dict)
    return ensemble_models


def combined_get_models_classification(train_answer, test_answer, answer_features, train_solver, test_solver,
                                       solver_features):
    models = combined_get_models()
    res = dict()
    X_train_answer = train_answer[answer_features]
    X_train_solver = train_solver[solver_features]
    y_train_answer = train_answer.Class
    y_train_solver = train_solver.Class
    X_test_answer = test_answer[answer_features]
    X_test_answer['Answer'] = test_answer['Answer']
    X_test_solver = test_solver[solver_features]
    X_test_solver['Answer'] = test_solver['Answer']

    for model in models.keys():
        models[model].fit_models(X_train_answer, y_train_answer, X_train_solver, y_train_solver)
        res['{}_prediction'.format(model)] = models[model].find_solution(X_test_answer, X_test_solver)

    return res


def combined_leave_one_out(df_answer, answer_features, df_solver, solver_features, bool_problem=False):
    final_res = {'correct_answer': [], 'Problem': [], 'Classifier': [], 'Classification': [], 'group_number': [],
                 'is_correct': []}
    groups = df_answer.group_number.unique()
    for group in groups:
        print('group {num} out of {all}'.format(num=group, all=len(groups)))
        problem = df_answer[df_answer.group_number == group].Problem.unique()[0]
        if bool_problem:
            train_df_answer = df_answer[df_answer.Problem != problem]
            train_df_solver = df_solver[df_solver.Problem != problem]
        else:
            train_df_answer = df_answer[df_answer.group_number != group]
            train_df_solver = df_solver[df_solver.group_number != group]
        test_df_answer = df_answer[df_answer.group_number == group]
        test_df_solver = df_solver[df_solver.group_number == group]
        res = combined_get_models_classification(train_df_answer, test_df_answer, answer_features, train_df_solver,
                                                 test_df_solver, solver_features)

        for key in res.keys():
            final_res['Problem'].append(problem)
            final_res['group_number'].append(group)
            correct = test_df_answer[test_df_answer.Class == 1].iloc[0]['Answer']
            final_res['correct_answer'].append(correct)
            final_res['Classifier'].append(key)
            final_res['Classification'].append(res[key])

            if res[key] == correct:
                final_res['is_correct'].append(1)
            else:
                final_res['is_correct'].append(0)

    results = pd.DataFrame(data=final_res)
    return results


def hpt(df, features):
    params = {'n_estimators': [i for i in range(100, 300, 50)], 'max_depth': [i for i in range(2, 11)],
              'min_samples_leaf': [i for i in range(2, 11)], 'class_weight': [{True: i, False: 1} for i in range(12)]}
    tester = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params)
    X = df[features]
    y = df['Class']
    tester.fit(X, y)
    print(tester.cv_results_)
    print(tester.best_params_)
    print(tester.get_params())


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        ans_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'featured data', 'answers_features.csv'))
        ans_df.fillna(-1, inplace=True)
        ans_df = ans_df[ans_df.AvgConf > 0]
        sol_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'featured data', 'subjects_features.csv'))
        sol_df.fillna(-1, inplace=True)
        sol_df = sol_df[sol_df.Confidence > 0]
        # hpt(sol_df, SOLVER_FEATURES)
        df = combined_leave_one_out(ans_df, ANSWER_FEATURES, sol_df, SOLVER_FEATURES)
        df.to_csv('ensemble_paramtuned.csv')
