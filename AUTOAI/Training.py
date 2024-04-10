import datetime
import json
import os
import shutil
import sklearn.base
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier as gbc
from xgboost.sklearn import XGBClassifier as xgb
import lightgbm as ltb
from sklearn import svm
from sklearn.model_selection import StratifiedKFold as skf
import configparser
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut 
from mlxtend.feature_selection.sequential_feature_selector import SequentialFeatureSelector
from mlxtend.feature_selection.sequential_feature_selector import _calc_score, _get_featurenames
import joblib
from joblib import Parallel, delayed
import numpy as np


global now_date_time
now_date_time = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")


class MySequentialFeatureSelector(SequentialFeatureSelector):

    def __init__(self, estimator, k_features=1, forward=True, floating=False, verbose=0, scoring=None, cv=5, n_jobs=1,
                 pre_dispatch='2*n_jobs', clone_estimator=True, fixed_features=None):

        super(MySequentialFeatureSelector, self).__init__(estimator, k_features, forward, floating, verbose, scoring,
                                                          cv, n_jobs, pre_dispatch, clone_estimator, fixed_features)

        self.all_results_ls = []

    def _inclusion(self, orig_set, subset, X, y, ignore_feature=None,
                   **fit_params):

        all_result_dt = dict()
        all_avg_scores = []
        all_cv_scores = []
        all_subsets = []
        res = (None, None, None)
        remaining = orig_set - subset
        if remaining:
            features = len(remaining)
            n_jobs = min(self.n_jobs, features)
            parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                                pre_dispatch=self.pre_dispatch)
            work = parallel(delayed(_calc_score)
                            (self, X, y,
                             tuple(subset | {feature}),
                             **fit_params)
                            for feature in remaining
                            if feature != ignore_feature)

            for new_subset, cv_scores in work:
                all_avg_scores.append(np.nanmean(cv_scores))
                all_cv_scores.append(cv_scores)
                all_subsets.append(new_subset)

            all_result_dt["avg_scores"] = all_avg_scores
            all_result_dt["cv_scores"] = all_cv_scores
            all_result_dt["idx"] = all_subsets

            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])

            self.all_results_ls.append(all_result_dt)

        return res


class Training:

    def __init__(self, hyper_parameters_json_file_pth=None, models="all", save_trained_model=False, output_file_name="", run_name="",
                  output_folder_pth="./path_to_outputs/"):

        self.save_model = save_trained_model
        self.run_date = now_date_time
        self.run_name = run_name
        
        _file_name = "{}_{}".format(self.run_date, output_file_name)
        
        self.save_pth = "{}{}/Save_model_{}/".format(output_folder_pth, _file_name, self.run_name)
        self.output_pth = "{}{}/".format(output_folder_pth, _file_name)

        self.selected_features = {}
        self.selected_features_score = {}

        self.predict_probabilities_dt = dict()
        self.gt_dt = dict()

        self.predict_probabilities_dt["train"] = {}
        self.predict_probabilities_dt["valid"] = {}
        self.predict_probabilities_dt["test"] = {}
        self.gt_dt["train"] = {}
        self.gt_dt["valid"] = {}
        self.gt_dt["test"] = {}

        grid_param_file = open("./grid_search_dt.json")
        self.grid_search_dt = json.load(grid_param_file)
        grid_param_file.close()

        if models == "all":
            self.models = {"C4.5 DT": tree.DecisionTreeClassifier, "RF": RandomForestClassifier,
                           "KNN": KNeighborsClassifier, "LogReg": LogisticRegression,
                           "AdaBoost": AdaBoostClassifier, "GBM": gbc, "LightGBM": ltb.LGBMClassifier,
                            "SVM": svm.SVC}

        else:
            self.models = models

        for model_name in self.models.keys():
            self.predict_probabilities_dt["train"][model_name] = []
            self.predict_probabilities_dt["valid"][model_name] = []
            self.predict_probabilities_dt["test"][model_name] = []
            self.gt_dt["train"][model_name] = []
            self.gt_dt["valid"][model_name] = []
            self.gt_dt["test"][model_name] = []

        if hyper_parameters_json_file_pth:
            print("Loading params from: {}".format(hyper_parameters_json_file_pth))
            with open(hyper_parameters_json_file_pth, "r") as j:
                self.hyper_parameters_dt = json.load(j)
                self.hyper_parameters_dt["AdaBoost"] = {"base_estimator": tree.DecisionTreeClassifier(**self.hyper_parameters_dt["C4.5 DT"])}
            j.close()
        else:
            print("Loading default params")
            with open("./default_param.json", "r") as j:
                self.hyper_parameters_dt = json.load(j)
                self.hyper_parameters_dt["AdaBoost"] = {"base_estimator": tree.DecisionTreeClassifier(**self.hyper_parameters_dt["C4.5 DT"])}
            j.close()

    def train(self, x, y, x_valid=None, y_valid=None, x_test=None, y_test=None, features=None, verbose=1, fold=0):

        for clf_name, ml_model in self.models.items():

            if features is not None and not self.selected_features:
                s_x = x[:, features[clf_name]]
                if x_valid is not None:
                    s_x_valid = x_valid[:, features[clf_name]]
                if x_test is not None:
                    s_x_test = x_test[:, features[clf_name]]

            elif self.selected_features:
                s_x = x[:, self.selected_features[clf_name]]
                if x_valid is not None:
                    s_x_valid = x_valid[:, self.selected_features[clf_name]]
                if x_test is not None:
                    s_x_test = x_test[:, self.selected_features[clf_name]]

            else:
                s_x = x
                s_x_valid = x_valid
                s_x_test = x_test
                
            if verbose:
                print('Now Training {} clf'.format(clf_name))

            clf = sklearn.base.clone(ml_model(**self.hyper_parameters_dt[clf_name]))
            clf.fit(s_x, y)

            if self.save_model:
                file_pth = "{}{}_{}.joblib".format(self.save_pth,
                                                      fold,
                                                      clf_name)
                joblib.dump(clf, file_pth)
                

            self.predict_probabilities_dt["train"][clf_name].append(clf.predict_proba(s_x))
            self.gt_dt["train"][clf_name].append(y)

            if x_valid is not None:
                self.predict_probabilities_dt["valid"][clf_name].append(clf.predict_proba(s_x_valid))
                self.gt_dt["valid"][clf_name].append(y_valid)

            if x_test is not None:
                self.predict_probabilities_dt["test"][clf_name].append(clf.predict_proba(s_x_test))
                self.gt_dt["test"][clf_name].append(y_test)

            del clf

        return self

    def grid_search(self, x, y, verbose=3, update_file_pth=None, cv=10):
        best_dt = {}

        for clf_name, ml_model in self.models.items():
            t1 = datetime.datetime.now()
            if verbose:
                print("currently running {} clf.".format(clf_name))
            clf = GridSearchCV(sklearn.base.clone(ml_model()), self.grid_search_dt[clf_name],
                               verbose=verbose, n_jobs=-1, cv=cv)
            clf.fit(x, y)
            best_dt[clf_name] = clf.best_params_
            t2 = datetime.datetime.now()
            print("Time taken for grid search {}:   {}".format(clf_name, str(t2-t1)))

        file_pth = "{}best_grid_search_params.json".format(self.output_pth)
        with open(file_pth, "w") as json_file:
            json.dump(best_dt, json_file)
            json_file.close()

        with open("./best_params.json", "r") as j:
            self.hyper_parameters_dt = json.load(j)
            self.hyper_parameters_dt["AdaBoost"] = {"base_estimator": tree.DecisionTreeClassifier(**self.hyper_parameters_dt["C4.5 DT"])}
        
        j.close()

        return best_dt

    def sequential_features_selection(self, x, y, verbose=1, features_name_ls=None, scoring="accuracy", cv=10,
                                      k_features="best"):

        for clf_name, model in self.models.items():
            t1 = datetime.datetime.now()

            if verbose:
                print("currently running {} clf.".format(clf_name))

            clf = model(**self.hyper_parameters_dt[clf_name])

            sfs = MySequentialFeatureSelector(clf, k_features=k_features, cv=cv, n_jobs=-1,
                                            verbose=verbose, scoring=scoring)

            sfs.fit(x, y)

            self.selected_features[clf_name] = sfs.k_feature_idx_
            self.selected_features_score[clf_name] = sfs.k_score_

            dis_pth = "{}FeatureSelection_{}/{}.joblib".format(self.output_pth,
                                                               self.run_name,
                                                               clf_name)

            print(dis_pth)
            joblib.dump(sfs, dis_pth)
            del sfs

            t2 = datetime.datetime.now()
            print("Time taken for SFS {}:   {}".format(clf_name, str(t2-t1)))

        return self

    def get_clf(self):
        get_clf_dt = {}
        for clf_name, model in self.models.items():
            clf = model(**self.hyper_parameters_dt[clf_name])
            get_clf_dt[clf_name] = clf
        return get_clf_dt


def run_train(x, y, verbose=1, update_params=True, update_file_pth=None, hyper_parameters_json_file_pth=None,
              grid_search=True, load_best_params=False, feature_selection=True, features=None, cv=10,
              k_features="best", models="all", split_test=False, x_test=None, y_test=None, output_file_name="", run_name="", save_trained_model=False, loo=False):
    
    idx_dt = {"train": [], "valid": [], "test": []}

    t = Training(hyper_parameters_json_file_pth=hyper_parameters_json_file_pth,
                load_best_params=load_best_params,
                models=models, output_file_name=output_file_name,
                run_name=run_name, save_trained_model=save_trained_model)

    if grid_search:
        t.grid_search(x, y, update_params=update_params, verbose=verbose,
                      update_file_pth=update_file_pth, cv=cv)

    if feature_selection:
        t.sequential_features_selection(x, y, verbose=verbose, cv=cv,
                                        k_features=k_features)
 
    if split_test and x_test is None:
        train_idx, test_idx, y_train, y_test = sklearn.model_selection.train_test_split(np.arange(0, x.shape[0]), y,
                                                                                        test_size=0.2,
                                                                                        random_state=10, stratify=y)
        x_train = x[train_idx, :]
        x_test = x[test_idx, :]
        idx_dt["test"].append(test_idx)
        
        if loo:
            model_selector = LeaveOneOut().split(x_train)
        else:
            model_selector = skf(n_splits=cv).split(x_train, y_train)
            
        for f, (t_idx, valid_idx) in enumerate(model_selector):
            t1 = datetime.datetime.now()
            idx_dt["train"].append(t_idx)
            idx_dt["valid"].append(valid_idx)
            train_x = x[t_idx, :]
            train_y = y[t_idx]
            valid_x = x[valid_idx, :]
            valid_y = y[valid_idx]
            t.train(train_x, train_y, x_valid=valid_x, y_valid=valid_y, x_test=x_test, y_test=y_test, features=features, fold=f)
            t2 = datetime.datetime.now()
            print("Time required for training {} fold: {}".format(f, str(t2-t1)))

    else:
        if loo:
            model_selector = LeaveOneOut().split(x)
        else:
            model_selector = skf(n_splits=cv).split(x, y)
            
        for f, (t_idx, valid_idx) in enumerate(model_selector):
            t1 = datetime.datetime.now()
            idx_dt["train"].append(t_idx)
            idx_dt["valid"].append(valid_idx)
            train_x = x[t_idx, :]
            train_y = y[t_idx]
            valid_x = x[valid_idx, :]
            valid_y = y[valid_idx]
            t.train(train_x, train_y, x_valid=valid_x, y_valid=valid_y, x_test=x_test, y_test=y_test, features=features, fold=f)
            t2 = datetime.datetime.now()
            print("Time required for training {} fold: {}".format(f, str(t2-t1)))

    out_dt = t.predict_probabilities_dt
    gt_dt = t.gt_dt

    if feature_selection:
        return t.selected_features, t.selected_features_score, out_dt, gt_dt, idx_dt
    else:
        return out_dt, gt_dt, idx_dt

    del t