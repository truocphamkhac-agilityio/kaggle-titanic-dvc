#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
# ======================================================================

import os

import pandas as pd
import yaml


def load_params(filepath="params.yaml") -> dict:
    """Helper function to load params.yaml

    Args:
        filepath (str): filename or full filepath to yaml file with parameters

    Returns:
        dict: dictionary of parameters
    """

    assert (os.path.isfile(filepath)), FileNotFoundError

    # read params.yaml
    with open(filepath, "r") as file:
        params = yaml.safe_load(file)

    return params


def convert_none_to_null(params):
    """Convert None values in params.yaml into null to ensure
    correct reading/writing as None type"""
    if isinstance(params, list):
        params[:] = [convert_none_to_null(elem) for elem in params]
    elif isinstance(params, dict):
        for k, v in params.items():
            params[k] = convert_none_to_null(v)
    return 'null' if params is None else params


def save_params(params):
    """"""
    # convert None values to null

    # save params
    new_params = yaml.safe_dump(params)

    with open("params.yaml", 'w') as writer:
        writer.write(new_params)


def load_data(data_path,
              sep=",",
              header=None,
              index_col=None) -> object:
    """Helper function to load train and test files
     as well as optional param loading

    Args:
        data_path (str or list of str): path to csv file
        sep (str):
        index_col (str):
        header (int):

    Returns:
        object:
    """

    # if single path as str, convert to list of str
    if type(data_path) is str:
        data_path = [data_path]

    # loop over filepath in list and read file
    output_df = [pd.read_csv(elem, sep=sep, header=header,
                                     index_col=index_col) for elem in data_path]
    # if single file as input, return single df not a list
    if len(output_df) == 1:
        output_df = output_df[0]

    return output_df


def save_as_csv(df, filepath, output_dir,
                replace_text=".csv",
                suffix="_processed.csv",
                na_rep="nan",
                output_path=False):
    """Helper function to format the new filename and save output"""

    # if single path as str, convert to list of str

    if type(df) is not list:
        df = [df]

    if type(filepath) is str:
        filepath = [filepath]

    # list lengths must be equal
    assert (len(df) == len(filepath)), AssertionError

    for temp_df, temp_path in zip(df, filepath):
        # set output filenames
        save_fname = os.path.basename(temp_path.replace(replace_text,
                                                        suffix))

        # save updated dataframes
        save_filepath = output_dir.joinpath(save_fname)
        temp_df.to_csv(save_filepath,
                       na_rep=na_rep)
        if output_path:
            return save_filepath
