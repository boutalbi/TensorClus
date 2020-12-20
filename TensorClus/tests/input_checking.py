# -*- coding: utf-8 -*-

"""

The :mod:`TensorClus.io.input_checking` module provides functions to check
input tensors.
"""

# Author: Rafika Boutalbi <rafika.boutalbi@gmail.com>
#         Mohamed Nadif <mohamed.nadif@u-paris.fr>
#         Lazhar Labiod <lazhar.labiod@u-paris.fr>

# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from scipy.sparse.sputils import isdense
from scipy.sparse.dok import dok_matrix
from scipy.sparse.lil import lil_matrix


def check_array(a, pos=True):
    """Check if an array contains numeric values with non empty rows nor
    columns.

    Parameters
    ----------
    a:
        The input tensor array
    pos: bool
        If ``True``, check if the values are positives
    Raises
    ------
    TypeError
        If the array is not a Numpy array or matrix or if the values are
        not numeric.
    ValueError
        If the array contains empty rows or columns or contains NaN values, or
        negative values (if ``pos`` is ``True``).
    """

    if not (type(a) == np.ndarray or type(a) == np.matrix):
        raise TypeError("Input data must be a Numpy/SciPy array or matrix")

    if ((not np.issubdtype(a.dtype.type, np.integer)) and (not np.issubdtype(a.dtype.type, np.floating))):
        raise TypeError("Input array or matrix must be of a numeric type")


    a = np.asarray(a)

    if len(np.where(~a.any(axis=0))[0]) > 0:
        raise ValueError("Zero-valued columns in data")
    if len(np.where(~a.any(axis=1))[1]) > 0:
        raise ValueError("Zero-valued rows in data")
    if pos:
        if (a < 0).any():
            raise ValueError("Negative values in data")
    if np.isnan(a).any():
        raise ValueError("NaN in data")


def check_positive(X):
    """Check if all values are positives.

    Parameters
    ----------
    X: numpy array or  matrix tensor
        Matrix to be analyzed
    Raises
    ------
    ValueError
        If the matrix contains negative values.
    Returns
    -------
    numpy array or scipy sparse matrix
        X
    """
    if isinstance(X, dok_matrix):
        values = np.array(list(X.values()))
    elif isinstance(X, lil_matrix):
        values = np.array([v for e in X.data for v in e])
    elif isdense(X):
        values = X
    else:
        values = X.data

    if (values < 0).any():
        raise ValueError("The matrix contains negative values.")

    return X


def check_numbers(matrix, n_clusters):
    """Check if the given matrix has enough rows and columns for the given
    number of co-clusters.

    Parameters
    ----------
    matrix:
        The input matrix
    n_clusters: int
        Number of co-clusters
    Raises
    ------
    ValueError
        If the data matrix has not enough rows or columns.
    """

    if matrix.shape[0] < n_clusters or matrix.shape[1] < n_clusters:
        raise ValueError("data matrix has not enough rows or columns")


def check_numbers_non_diago(matrix, n_row_clusters, n_col_clusters):
    """Check if the given matrix has enough rows and columns for the given
    number of row and column clusters.

    Parameters
    ----------
    matrix:
        The input matrix
    n_row_clusters: int
        Number of row clusters
    n_col_clusters: int
        Number of column clusters
    Raises
    ------
    ValueError
        If the data matrix has not enough rows or columns.
    """

    if matrix.shape[0] < n_row_clusters or matrix.shape[1] < n_col_clusters:
        raise ValueError("data matrix has not enough rows or columns")


def check_numbers_clustering(matrix, n_clusters):
    """Check if the given matrix has enough rows and columns for the given
    number of clusters.

    Parameters
    ----------
    matrix:
        The input matrix
    n_clusters: int
        Number of clusters
    Raises
    ------
    ValueError
        If the data matrix has not enough rows or columns.
    """

    if matrix.shape[0] < n_clusters:
        raise ValueError("data matrix has not enough rows")



def check_tensor(a, pos=True):
    """Check if an array is a tensor

    Parameters
    ----------
    a:
        The input tensor array
    pos: bool
        If ``True``, check if the values are positives
    Raises
    ------
    TypeError
        If the array is not a Numpy array or matrix or if the values are
        not numeric.
    ValueError
        If the array contains empty rows or columns or contains NaN values, or
        negative values (if ``pos`` is ``True``).
    """

    if not (type(a) == np.ndarray ):
        raise TypeError("Input data must be a Numpy array or matrix")

    if ((not np.issubdtype(a.dtype.type, np.integer)) and (not np.issubdtype(a.dtype.type, np.floating))):
        raise TypeError("Input array or matrix must be of a numeric type")


    a = np.asarray(a)


    if a.shape[2]  <0:
        raise ValueError("Null third dimension")
    if pos:
        if (a < 0).any():
            raise ValueError("Negative values in data")
    if np.isnan(a).any():
        raise ValueError("NaN in data")


def check_positive_tensor(X):
    """Check if all values are positives.

    Parameters
    ----------
    X: three-way numpy array
        tensor to be analyzed
    Raises
    ------
    ValueError
        If the tensor contains negative values.
    Returns
    -------
    three-way numpy array
        X
    """

    if (X < 0).any():
        raise ValueError("The matrix contains negative values.")

    return X


def check_numbers_clusters(tensor, n_clusters):
    """Check if the given matrix has enough rows and columns for the given
    number of co-clusters.

    Parameters
    ----------
    tensor:
        The input tensor
    n_clusters: int
        Number of co-clusters
    Raises
    ------
    ValueError
        If the data matrix has not enough rows or columns.
    """

    if tensor.shape[0] < n_clusters or tensor.shape[1] < n_clusters:
        raise ValueError("data matrix has not enough rows or columns")


def check_numbers_clusters_non_diago(tensor, n_row_clusters, n_col_clusters):
    """Check if the given matrix has enough rows and columns for the given
    number of row and column clusters.

    Parameters
    ----------
    tensor:
        The input tensor
    n_row_clusters: int
        Number of row clusters
    n_col_clusters: int
        Number of column clusters
    Raises
    ------
    ValueError
        If the data matrix has not enough rows or columns.
    """

    if tensor.shape[0] < n_row_clusters or tensor.shape[1] < n_col_clusters:
        raise ValueError("data matrix has not enough rows or columns")
