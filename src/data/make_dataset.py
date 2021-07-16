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

from kaggle.api.kaggle_api_extended import KaggleApi

from src.data import data_dictionary


def download_data(competition, train_data, test_data,
                  output_dir="./data/raw",
                  credentials=".kaggle/kaggle.json"):
    """Download raw dataset from Kaggle"""
    credentials = Path.home().joinpath(credentials)
    output_dir = Path(output_dir).resolve()

    assert (os.path.isfile(credentials)), FileNotFoundError(credentials)
    assert (os.path.isdir(output_dir)), NotADirectoryError(output_dir)

    api = KaggleApi()
    api.authenticate()

    # downloading from kaggle.com/c/titanic
    api.competition_download_file(competition,
                                  train_data, path=output_dir)
    api.competition_download_file(competition,
                                  test_data, path=output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--competition", dest="competition",
                        required=True, help="Kaggle competition to download")
    parser.add_argument("-tr", "--train_data", dest="train_data",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test_data", dest="test_data",
                        required=True, help="Test CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=os.path.dirname(Path(__file__).resolve()),
                        required=False, help="output directory")
    args = parser.parse_args()

    # set vars
    args.output_dir = Path(args.output_dir).resolve()
    train_path = args.output_dir.joinpath(args.train_data)
    test_path = args.output_dir.joinpath(args.test_data)

    # download dataset from kaggle
    download_data(args.competition, args.train_data, args.test_data,
                  output_dir=args.output_dir)

    data_dictionary.create(train_path)
