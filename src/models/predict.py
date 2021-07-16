#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd

from src.data import load_data, load_params, save_as_csv
from src.models.metrics import james_stein


def main(test_path, results_dir, model_dir,
         model_name="estimator.pkl"):
    """Predict survival on held-out test dataset"""

    assert (os.path.isdir(results_dir)), NotADirectoryError
    assert (os.path.isdir(model_dir)), NotADirectoryError
    results_dir = Path(results_dir).resolve()
    model_dir = Path(model_dir).resolve()

    # load estimator
    model_filepath = model_dir.joinpath(model_name)
    assert (os.path.isfile(model_filepath)), FileNotFoundError
    with open(model_filepath, 'rb') as model_file:
        cv_estimators = pickle.load(model_file)

    # read test df
    test_df = load_data(test_path,
                        sep=",", header=0,
                        index_col="PassengerId")

    # load params
    params = load_params()
    target_class = params["train_test_split"]["target_class"]
    js_estimator = params["predict"]["js_estimator"]

    # get independent variables (features) and
    # dependent variables (labels)
    if target_class in test_df.columns:
        test_feats = test_df.drop(target_class, axis=1)
    else:
        test_feats = test_df

    # predict output
    output = [model.predict_proba(test_feats)[:, 1] for model in cv_estimators]

    # create df
    output_df = pd.DataFrame(output).transpose().set_index(test_feats.index)

    if js_estimator:
        # compute James-Stein estimate for the mean of N-fold cross-validation
        p_hat_js = james_stein(output_df, limit_shrinkage=True)
        output_proba = p_hat_js.rename(columns={0: target_class})
    else:
        output_proba = pd.DataFrame(output_df.mean(axis=1)).rename(columns={0: target_class})

    # binarize
    output_binary = (output_proba > 0.5).astype(int)

    # save output
    save_as_csv(output_proba, test_path, results_dir,
                replace_text="_processed.csv",
                suffix="_predict_proba.csv",
                na_rep="nan")
    save_as_csv(output_binary, test_path, results_dir,
                replace_text="_processed.csv",
                suffix="_predict_binary.csv",
                na_rep="nan")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-te", "--test", dest="test_path",
                        required=True, help="CSV file")
    parser.add_argument("-rd", "--results-dir", dest="results_dir",
                        default=Path("./results").resolve(),
                        required=False, help="Metrics output directory")
    parser.add_argument("-md", "--model-dir", dest="model_dir",
                        default=Path("./models").resolve(),
                        required=False, help="Model output directory")
    args = parser.parse_args()

    # train model
    main(args.test_path, args.results_dir, args.model_dir)
