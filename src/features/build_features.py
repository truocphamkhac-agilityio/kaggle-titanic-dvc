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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from src.data import load_data, load_params, save_as_csv


def main(train_path, test_path,
         output_dir):
    """Build features
    TODO- Currently a placeholder script that saves existing files until feature engineering is implemented"""

    output_dir = Path(output_dir).resolve()
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # load train and test data because feature engineering process should be identical
    train_df, test_df = load_data([train_path, test_path],
                                  sep=",", header=0,
                                  index_col="PassengerId")

    params = load_params()
    target_class = params["train_test_split"]["target_class"]

    # pop the target class
    train_labels = train_df.pop(target_class)

    # concatenate df
    df = pd.concat([train_df, test_df], sort=False)

    # load params
    params = load_params()
    params_featurize = params["feature_eng"]
    params_featurize["random_seed"] = params["random_seed"]

    # optionally normalize data
    if params_featurize["featurize"]:
        # create poly features
        df = create_poly_features(df, degree=2,
                                  interaction_only=True)

        # hand-crafted features
        df = hand_crafted_features(df)

        # bin continuous features
        df['Age'] = pd.qcut(df['Age'], 10,
                            duplicates="drop").astype('category').cat.codes
        df['Fare'] = pd.qcut(df['Fare'], 13).astype('category').cat.codes
        df['family_size'] = pd.qcut(df['family_size'], 3,
                                    duplicates="drop").astype('category').cat.codes

    # return datasets to train and test
    train_df = df.loc[train_df.index, df.columns]
    train_df.insert(loc=0, column=target_class,
                    value=train_labels)
    test_df = df.loc[test_df.index, df.columns]

    # save data
    save_as_csv([train_df, test_df],
                [train_path, test_path],
                output_dir,
                replace_text="_nan_imputed.csv",
                suffix="_featurized.csv",
                na_rep="nan")


def hand_crafted_features(df):
    df["family_size"] = df["SibSp"] + df["Parch"] +1
    df["is_vip"] = is_vip(df)
    df["parent"] = is_parent(df)
    df["is_orphan"] = is_orphan(df)
    df["is_single_adult_mother"] = is_single_adult_mother(df)
    df["is_single_adult_male"] = is_single_adult_male(df)
    return df

def is_vip(df):
    return pd.DataFrame([df["Pclass"] == 1,
                         df["Fare"] > np.percentile(df["Fare"], 95)]).transpose().all(axis=1).astype(int)


def is_parent(df):
    return pd.DataFrame([df["Parch"] == 1,
                         df["Age"] >= 18]).transpose().all(axis=1).astype(int)


def is_orphan(df):
    return pd.DataFrame([df["Parch"] == 0,
                         df["SibSp"] == 0,
                         df["Age"] < 18]).transpose().all(axis=1).astype(int)


def is_single_adult_mother(df):
    return pd.DataFrame([df["Parch"] > 0,
                         df["SibSp"] == 0,
                         df["Sex"] == 0,
                         df["Age"] >= 18]).transpose().all(axis=1).astype(int)


def is_single_adult_male(df):
    return pd.DataFrame([df["Parch"] == 0,
                         df["SibSp"] == 0,
                         df["Sex"] == 1,
                         df["Age"] >= 18]).transpose().all(axis=1).astype(int)

def create_poly_features(df, degree=2,
                         interaction_only=True):
    # create polynomial feature instance
    poly = PolynomialFeatures(degree=degree,
                              interaction_only=interaction_only)
    poly.fit_transform(df.to_numpy())
    poly_cols = poly.get_feature_names(df.columns)
    poly_df = pd.DataFrame(poly.fit_transform(df.to_numpy()),
                           columns=poly_cols).set_index(df.index)
    return poly_df.drop(columns=poly_cols[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test", dest="test_path",
                        required=True, help="Test CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=Path("./data/interim").resolve(),
                        required=False, help="output directory")
    args = parser.parse_args()

    # convert categorical variables into integer codes
    main(args.train_path, args.test_path,
         args.output_dir)
