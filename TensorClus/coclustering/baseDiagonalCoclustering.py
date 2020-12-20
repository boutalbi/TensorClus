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


class BaseDiagonalCoclust(BaseEstimator):
    def get_indices(self, i):
        """Give the row and column indices of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the co-cluster
        Returns
        -------
        (list, list)
            (row indices, column indices)
        """
        row_indices = [index for index, label in enumerate(self.row_labels_)
                       if label == i]
        column_indices = [index for index, label
                          in enumerate(self.column_labels_) if label == i]
        return (row_indices, column_indices)

    def get_shape(self, i):
        """Give the shape of the i’th co-cluster.

        Parameters
        ----------
        i : integer
            Index of the co-cluster
        Returns
        -------
        (int, int)
            (number of rows, number of columns)
        """
        row_indices, column_indices = self.get_indices(i)
        return (len(row_indices), len(column_indices))

    def get_submatrix(self, m, i):
        """Give the submatrix corresponding to row cluster i and column cluster j.

        Parameters
        ----------
        m : X : numpy three-way array
            Matrix from which the block has to be extracted
        i : integer
           index of the row cluster
        Returns
        -------
        numpy array or scipy sparse matrix
            Submatrix corresponding to row cluster i and column cluster j
        """
        row_ind, col_ind = self.get_indices(i)
        row_ind = np.asarray(row_ind).reshape(m.shpae[0])
        col_ind = np.asarray(col_ind).reshape(m.shpae[1])
        m_= m[row_ind,:,:]
        m_= m[:,col_ind,:]
        return m_


