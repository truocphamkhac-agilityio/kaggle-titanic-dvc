#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def gmpr_score(y_true, y_pred, weights=None):
    """Compute the geometric mean of precision and recall"""
    # TODO - compare with sklearn FM index
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')

    # update weights parameter and check attributes
    weights = [0.5, 0.5] if weights is None else weights
    assert (type(weights) is list), TypeError
    assert (len(weights) == 2), TypeError

    # compute geometric mean (equally weighted by class)
    gmpr = np.product(np.power([precision, recall], weights))

    return gmpr


def james_stein(df, limit_shrinkage=True):
    """James-Stein estimator for predictions"""
    assert (type(df) is type(pd.DataFrame())), TypeError

    # save n_cols
    n_rows, n_cols = df.shape
    if n_cols > 1:
        df = df.mean(axis=1)

    # compute the grand mean
    p_hat = df.mean(axis=0)
    sigma2 = np.divide(p_hat * (1 - p_hat), n_cols)  # binomial variance
    mle_diff = df - p_hat
    sum_sq_errors = np.sum(np.square(mle_diff))
    shrinkage = 1 - np.divide((n_rows - 3) * sigma2, sum_sq_errors)
    p_hat_js = p_hat + np.multiply(shrinkage, mle_diff)

    # limited translation of James-Stein, which does not allow
    # JS estimate to diverge more than one sigma from p_hat
    if limit_shrinkage:
        js_upper = np.amax([p_hat_js,
                            (df - np.sqrt(sigma2))], axis=0)
        p_hat_js = np.amin([js_upper,
                            (df + np.sqrt(sigma2))], axis=0)

    # create output DataFrame
    p_hat_js = pd.DataFrame(p_hat_js).set_index(df.index)

    return p_hat_js
