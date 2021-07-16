#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

from src.data import load_data, load_params
from src.models.metrics import gmpr_score


def main(train_path, cv_idx_path,
         results_dir, model_dir):
    """Train RandomForest model and predict survival on
    Kaggle test set"""
    assert (os.path.isdir(results_dir)), NotADirectoryError
    assert (os.path.isdir(model_dir)), NotADirectoryError
    results_dir = Path(results_dir).resolve()
    model_dir = Path(model_dir).resolve()

    # read files
    train_df, cv_idx = load_data([train_path, cv_idx_path],
                                 sep=",", header=0,
                                 index_col="PassengerId")
    # load params
    params = load_params()
    classifier = params["classifier"]
    target_class = params["train_test_split"]["target_class"]
    model_params = params["model_params"][classifier]

    # get independent variables (features) and
    # dependent variables (labels)
    train_feats = train_df.drop(target_class, axis=1)
    train_labels = train_df[target_class]

    # create instance using random seed for reproducibility
    if classifier.lower() == "random_forest":
        model = RandomForestClassifier(**model_params,
                                       random_state=params["random_seed"])
    elif classifier.lower() == "xgboost":
        model = XGBClassifier(random_state=params["random_seed"])
    else:
        raise NotImplementedError

    # create generator with cv splits
    split_generator = iter((np.where(cv_idx[col] == "train")[0],
                            np.where(cv_idx[col] == "test")[0]) for col in cv_idx)

    # set model scoring metrics
    # TODO - add custom metric for GMPR
    scoring = {'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy',
               'f1': 'f1',
               "gmpr": make_scorer(gmpr_score, greater_is_better=True),
               'jaccard': 'jaccard', 'precision': 'precision',
               'recall': 'recall', 'roc_auc': 'roc_auc'}

    # train using cross validation
    cv_output = cross_validate(model, train_feats.to_numpy(),
                               train_labels.to_numpy(),
                               cv=split_generator,
                               fit_params=None,
                               scoring=scoring,
                               return_estimator=True)

    # get cv estimators
    cv_estimators = cv_output.pop('estimator')
    cv_metrics = pd.DataFrame(cv_output)

    # rename columns
    col_mapper = dict(zip(cv_metrics.columns,
                          [elem.replace('test_', '') for elem in cv_metrics.columns]))
    cv_metrics = cv_metrics.rename(columns=col_mapper)

    # save cv estimators as pickle file
    with open(model_dir.joinpath("estimator.pkl"), "wb") as file:
        pickle.dump(cv_estimators, file)

    # save metrics
    metrics = json.dumps(dict(cv_metrics.mean()))
    with open(results_dir.joinpath("metrics.json"), "w") as writer:
        writer.writelines(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-cv", "--cvindex", dest="cv_index",
                        required=True, help="CSV file with train/dev split")
    parser.add_argument("-rd", "--results-dir", dest="results_dir",
                        default=Path("./results").resolve(),
                        required=False, help="Metrics output directory")
    parser.add_argument("-md", "--model-dir", dest="model_dir",
                        default=Path("./models").resolve(),
                        required=False, help="Model output directory")
    args = parser.parse_args()

    # train model
    main(args.train_path, args.cv_index,
         args.results_dir, args.model_dir)
