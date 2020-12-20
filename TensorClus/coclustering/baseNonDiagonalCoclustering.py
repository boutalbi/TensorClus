# -*- coding: utf-8 -*-

"""

The :mod:`TensorClus.coclustering.baseNonDiagonalCoclustering` module provides a
class with common methods for non diagonal tensor co-clustering algorithms.
"""

# Author: Rafika Boutalbi <rafika.boutalbi@gmail.com>
#         Mohamed Nadif <mohamed.nadif@u-paris.fr>
#         Lazhar Labiod <lazhar.labiod@u-paris.fr>

# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator


class BaseNonDiagonalCoclust(BaseEstimator):
    def get_row_indices(self, i):
        """Give the row indices of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the i'th row cluster
        Returns
        -------
        list
            list of row indices
        """
        row_indices = [index for index, label in enumerate(self.row_labels_)
                       if label == i]
        return row_indices

    def get_col_indices(self, i):
        """Give the column indices of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the i'th column cluster
        Returns
        -------
        list
            list of column indices
        """
        col_indices = [index for index, label in enumerate(self.column_labels_)
                       if label == i]
        return col_indices

    def get_shape(self, i, j):
        """Give the shape of block corresponding to the i’th row cluster and
           the j'th column cluster.

        Parameters
        ----------
        i : integer
            Index of the row cluster
        j : integer
            Index of the column cluster
        Returns
        -------
        (int, int)
            (number of rows, number of columns)
        """
        row_indices = self.get_row_indices(i)
        column_indices = self.get_col_indices(i)
        return (len(row_indices), len(column_indices))

    def get_submatrix(self, m, i, j):
        """Give the submatrix corresponding to row cluster i and column cluster j.

        Parameters
        ----------
        m : X : numpy three-way array
            Matrix from which the block has to be extracted
        i : integer
           index of the row cluster
        j : integer
           index of the col cluster
        Returns
        -------
        numpy array or scipy sparse matrix
            Submatrix corresponding to row cluster i and column cluster j
        """
        row_ind = np.array(self.get_row_indices(i)).reshape(m.shape[0])
        col_ind = np.array(self.get_col_indices(j)).reshape(m.shape[1])
        m_= m[row_ind,:,:]
        m_= m[:,col_ind,:]
        return m_


