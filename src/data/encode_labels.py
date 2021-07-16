#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
# ======================================================================

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml

from src.data import load_data, load_params, save_as_csv


def main(train_path, test_path,
         output_dir, remove_nan=False,
         label_dict_name="label_encoding.yaml"):
    """Encode categorical labels as numeric, save the processed
    dataset the label encoding dictionary"""

    output_dir = Path(output_dir).resolve()
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # load data
    train_df, test_df = load_data([train_path, test_path], sep=",", header=0,
                                  index_col="PassengerId")

    # load params
    params = load_params()

    # update params for column data types
    param_dtypes = params["dtypes"]
    param_dtypes["Pclass"] = pd.api.types.CategoricalDtype(categories=[1, 2, 3],
                                                           ordered=True)
    # concatenate df
    df = pd.concat([train_df, test_df], sort=False)
    df = df.astype(param_dtypes)

    # drop unnecessary columns
    df = df.drop(columns=params["drop_cols"])

    # convert to categorical
    encoding_dict = {}
    for elem, col in zip(df.dtypes, df.columns):
        if isinstance(elem, pd.CategoricalDtype):
            # save mapping of category to integer class
            encoding_dict[col] = {key: val for key, val in enumerate(elem.categories)}

            # transform to categorical codes
            df[col] = df[col].cat.codes

    # return datasets to train and test
    train_df = df.loc[train_df.index, df.columns]
    test_df = df.loc[test_df.index, df.columns[1:]]

    # remove nan (if applicable
    if remove_nan:
        train_df = train_df.dropna(axis=0, how="any")

    # save data
    save_as_csv([train_df, test_df],
                [train_path, test_path],
                output_dir,
                replace_text=".csv",
                suffix="_categorized.csv",
                na_rep="nan")

    # save and encoding dictionaries
    encoding_dict = yaml.safe_dump(encoding_dict)
    with open(os.path.join(output_dir, label_dict_name), "w") as writer:
        writer.writelines(encoding_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test", dest="test_path",
                        required=True, help="Test CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=Path("./data/interim").resolve(),
                        required=False, help="output directory")
    parser.add_argument("-r", "--remove-nan", dest="remove_nan",
                        default=False, required=False,
                        help="Remove nan rows from training dataset")
    parser.add_argument("-l", "--label", dest="label_dict_name",
                        default="label_encoding.yaml",
                        required=False, help="Name for dictionary mapping category codes to text")
    args = parser.parse_args()

    # convert categorical variables into integer codes
    main(args.train_path, args.test_path,
         args.output_dir,
         remove_nan=args.remove_nan,
         label_dict_name=args.label_dict_name)
