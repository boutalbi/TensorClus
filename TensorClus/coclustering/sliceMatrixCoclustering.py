# -*- coding: utf-8 -*-

"""

The :mod:`TensorClus.coclustering.sliceMatrixCoclustering` module provides an implementation
of a Sparse tensor co-clustering algorithm.
"""

# Author: Rafika Boutalbi <rafika.boutalbi@gmail.com>
#         Mohamed Nadif <mohamed.nadif@u-paris.fr>
#         Lazhar Labiod <lazhar.labiod@u-paris.fr>

# License: BSD 3 clause

from __future__ import division
import numpy as np
import coclust
from coclust.coclustering import CoclustMod
from coclust.coclustering import CoclustInfo
from coclust.coclustering import CoclustSpecMod

from sklearn.utils import check_random_state, check_array
from ..initialization import random_init


class SliceMatrixCoclustering():
    """Co-clustering of tensor's slices.

    Parameters
    ----------
    slices : array-like, optional, default: [1,2]
        Selected modes for clustering

    algorithms : string or instance form colust package optional, default: "kmeans++"
        Selected algorithm for coclustering
    Attributes
    ----------
    models_ : obtained models
        co-clustering results of selected algorithms
    """

    def __init__(self, slices = [1,2],algorithms = ["Kmeans++"], n_clusters = 3):
        self.slices = slices
        self.algorithms = algorithms
        self.n_clusters = n_clusters
        self.models_ = None

    def fit(self, X, y=None):
        """Perform slice co-clustering.

        Parameters
        ----------
        X : Tensor

        Returns
        -------
        list
            list of obtained models
        """
        slices = self.slices
        algorithms = self.algorithms
        n_clusters = self.n_clusters
        print("slices ", slices)
        listModels = []
        listModelsSlice =[]
        for s,slice in enumerate(slices):
            slice_ = slice - 1
            data_slice = X[:,:,slice_]
            data_slice = np.asarray(data_slice)
            for a, algorithm in enumerate(algorithms):
                if isinstance(algorithm, str):
                    if algorithm == "CoclustMod":
                        model = CoclustMod(n_clusters=n_clusters)
                        model.fit(data_slice)
                    elif algorithm == "CoclustInfo":
                        model = CoclustInfo(n_row_clusters=n_clusters, n_col_clusters=n_clusters)
                        model.fit(data_slice)
                    elif algorithm == "CoclustSpecMod":
                        model = CoclustSpecMod(n_clusters=n_clusters)
                        model.fit(data_slice)
                    else:
                        print("The selected algorithm does not exists")
                else:
                    model = algorithm
                    model.fit(data_slice)

                listModelsSlice.append(model)
            listModels.append(listModelsSlice)
        self.models_ = listModels
        return self
