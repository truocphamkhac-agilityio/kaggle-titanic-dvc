#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tableone import TableOne

from src.data import load_data, load_params


def create(data_path, report_dir="./reports/figures",
           output_file="data_dictionary.tex"):
    """Create a data dictionary"""
    assert (os.path.isfile(data_path)), FileNotFoundError
    assert (os.path.isdir(report_dir)), NotADirectoryError
    report_dir = Path(report_dir).resolve()

    # read files - do not specify index column
    df = pd.read_csv(data_path, sep=",", header=0,
                     na_values=["nan"])

    # load params
    params = load_params()

    # update params for column data types
    param_dtypes = params["dtypes"]
    param_dtypes["Pclass"] = pd.api.types.CategoricalDtype(categories=[1, 2, 3],
                                                           ordered=True)
    df = df.astype(param_dtypes)

    # Save information about column names, null count, and dtypes
    col_list = df.columns.to_list()
    n_cols = np.arange(0, len(col_list))
    total_rows = df.shape[0]
    null_count = total_rows - df.isna().sum()
    col_dtype = df.dtypes

    # additional processing for categorical
    category_list = []
    drange_list = []
    ordered_list = []
    for col, elem in zip(col_list, col_dtype.to_list()):
        if str(elem) in {"category"}:
            category_list.append(elem.categories.to_list())
            drange_list.append([df[col].cat.categories.min(),
                                df[col].cat.categories.max()])
            ordered_list.append(str(elem.ordered))
        elif str(elem) in {"float64", "float32",
                           "int64", "int64"}:
            category_list.append("")
            drange_list.append([round(df[col].min(), 2),
                                round(df[col].max(), 2)])
            ordered_list.append("")
        else:
            category_list.append("")
            drange_list.append("")
            ordered_list.append("")

    out_df = pd.DataFrame(data={"#": n_cols, "Column": col_list,
                                "Non-null count": null_count.to_numpy(),
                                "Dtype": col_dtype.to_list(),
                                "Drange": drange_list,
                                "Categories": category_list,
                                "Ordered": ordered_list})
    # write table to latex
    template = r'''\documentclass[preview]{{standalone}}
    \usepackage{{booktabs}}
    \begin{{document}}
    {}
    \end{{document}}
    '''

    output_file = report_dir.joinpath(output_file)
    with open(output_file, "w") as file:
        file.write(template.format(out_df.to_latex()))

    # convert tex to PDF
    if os.path.isfile(output_file) and sys.platform == "linux":
        subprocess.call(["pdflatex", "--output-directory",
                         report_dir, output_file])

    # create instance of tableone and save summary statistics
    summary_df = df.drop(columns=["PassengerId", "Cabin", "Embarked", "Ticket", "Name"])
    categorical_idx = summary_df.columns[summary_df.dtypes == "category"].to_list()
    sig_digits = {"Age": 1, "Fare": 2}
    min_max = ["Parch", "SibSp"]
    mytable = TableOne(summary_df,
                       columns=summary_df.columns.to_list(),
                       categorical=categorical_idx,
                       decimals=sig_digits,
                       min_max=min_max)

    # save table one
    # write table to latex
    table_filepath = report_dir.joinpath("table_one.tex")
    with open(table_filepath, "w") as file:
        file.write(template.format(mytable.to_latex()))

    # convert tex to PDF
    if os.path.isfile(table_filepath) and sys.platform == "linux":
        subprocess.call(["pdflatex", "--output-directory",
                         report_dir, table_filepath])
