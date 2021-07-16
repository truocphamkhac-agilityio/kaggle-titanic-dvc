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

from src.data import load_data, load_params, save_as_csv


def main(train_path, test_path,
         output_dir):
    """Normalize data"""

    output_dir = Path(output_dir).resolve()
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # set vars
    norm_method = {"min_max", "z_score"}

    # load data
    train_df, test_df = load_data([train_path, test_path], sep=",", header=0,
                                  index_col="PassengerId")

    # load params
    params = load_params()

    # optionally normalize data
    if params["normalize"] in norm_method:
        # TODO add function to normalize data
        raise NotImplementedError

    # save data
    save_as_csv([train_df, test_df],
                [train_path, test_path],
                output_dir,
                replace_text="_featurized.csv",
                suffix="_processed.csv",
                na_rep="nan")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test", dest="test_path",
                        required=True, help="Test CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=Path("./data/processed").resolve(),
                        required=False, help="output directory")
    args = parser.parse_args()

    # convert categorical variables into integer codes
    main(args.train_path, args.test_path,
         args.output_dir)
