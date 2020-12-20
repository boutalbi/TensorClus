# -*- coding: utf-8 -*-

"""

The :mod:`TensorClus.decomposition.decomposition_with_clustering` module provides a
class with common methods for multiple clustering alorihtm from decomposition results.
"""

# Author: Rafika Boutalbi <rafika.boutalbi@gmail.com>
#         Mohamed Nadif <mohamed.nadif@u-paris.fr>
#         Lazhar Labiod <lazhar.labiod@u-paris.fr>

# License: BSD 3 clause
from __future__ import division
import numpy as np
import scipy as sp
import random
from random import randint
from scipy import special


from sklearn.utils import check_random_state, check_array
from sklearn.cluster import KMeans
from coclust.clustering.spherical_kmeans import SphericalKmeans
from sklearn.cluster import SpectralClustering
from sklearn import mixture

from ..initialization import random_init
from sklearn.base import BaseEstimator
from TensorClus.tests.input_checking import check_positive


class DecompositionWithClustering(BaseEstimator):
    """Clustering from decomposition results.
    
    Parameters
    ----------
    n_clusters : array-like, optional, default: [2,2,2]
        Number of row clusters to form

    modes : array-like, optional, default: [1,2,3]
        Selected modes for clustering

    algorithm : string, optional, default: "kmeans++"
        Selected algorithm for clustering
    Attributes
    ----------
    labels_ : array-like, shape (n_rows,)
        clustering label of each row
    """

    def __init__(self, n_clusters = [2,2,2], modes = [1,2,3], algorithm = "Kmeans++"):
        self.n_clusters = n_clusters
        self.modes = modes
        self.algorithm = algorithm
        self.labels_ = None

    def fit(self, X, y=None):
        """Perform Tensor co-clustering.
        
        Parameters
        ----------
        X : decomposition results
        """

        n_clusters = self.n_clusters
        modes = self.modes
        algorithm = self.algorithm
        print("modes ", modes)
        listClustering = []
        for m,mode in enumerate(modes):
            mode_= mode - 1
            nbCluster = n_clusters[m]
            data_mode = X[mode_]
            data_mode = np.asarray(data_mode)
            if algorithm == "Kmeans++":
                kmeans = KMeans(n_clusters=nbCluster, random_state=0).fit(data_mode)
                listClustering.append(kmeans.labels_)
            elif algorithm == "Skmeans":
                SphCluster = SphericalKmeans(n_clusters=nbCluster, n_init=1)
                SphCluster.fit(data_mode + 1.e-9)
                listClustering.append(SphCluster.labels_)
            elif algorithm == "SpectralClustering":
                Specclustering = SpectralClustering(n_clusters=nbCluster,assign_labels = "discretize",random_state = 0).fit(data_mode)
                listClustering.append(Specclustering.labels_)
            elif algorithm == "GMM":
                gmm = mixture.GaussianMixture(n_components=nbCluster, covariance_type='full')
                ypred = gmm.fit_predict(data_mode)
                listClustering.append(ypred)

        self.labels_ = listClustering
        return self
