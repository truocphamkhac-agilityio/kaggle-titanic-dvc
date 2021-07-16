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

import yaml

from src.data import load_data, load_params, save_as_csv


def main(train_path, test_path,
         output_dir):
    """Split data into train, dev, and test"""

    output_dir = Path(output_dir).resolve()
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # load data
    train_df, test_df = load_data([train_path, test_path], sep=",", header=0,
                                  index_col="PassengerId")

    # load params
    params = load_params()

    # fill nans with column mean/mode on test set
    # TODO - switch to allow for different interpolation methods (e.g., mean, median, MICE)
    if params["imputation"]["method"].lower() == "mean":
        mean_age = float(round(train_df["Age"].mean(), 4))
        mean_fare = float(round(train_df["Fare"].mean(), 4))
        train_df["Age"].fillna(value=mean_age,
                               inplace=True)
        test_df["Age"].fillna(value=mean_age,
                              inplace=True)
        test_df["Fare"].fillna(value=mean_fare,
                              inplace=True)

        # update params and save imputation scheme
        params["imputation"]["Age"] = mean_age
        params["imputation"]["Fare"] = mean_fare
    elif params["imputation"]["method"].lower() == "mice":
        # TODO MICE interpolation
        raise NotImplementedError
    else:
        raise NotImplementedError

    # update params
    new_params = yaml.safe_dump(params)

    with open("params.yaml", "w") as writer:
        writer.write(new_params)

    # save data
    save_as_csv([train_df, test_df],
                [train_path, test_path],
                output_dir,
                replace_text="_categorized.csv",
                suffix="_nan_imputed.csv",
                na_rep="nan")


if __name__ == "__main__":
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
