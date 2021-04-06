# -*- coding: utf-8 -*-

"""

The :mod:`TensorClus.coclustering.tensorCoclusteringBernoulli` module provides an implementation
of a tensor co-clustering algorithm for binary three-way tensor.
"""

# Author: Rafika Boutalbi <rafika.boutalbi@gmail.com>
#         Mohamed Nadif <mohamed.nadif@u-paris.fr>
#         Lazhar Labiod <lazhar.labiod@u-paris.fr>

# License: BSD 3 clause

from __future__ import division
import numpy as np
from numpy import linalg
import scipy as sp
import random
from random import randint
from scipy import special
from numpy.linalg import inv

from sklearn.utils import check_random_state, check_array

from ..initialization import random_init
from .baseNonDiagonalCoclustering import BaseNonDiagonalCoclust
from ..tests.input_checking import check_tensor,check_numbers_clusters_non_diago

GPU_exist = False
try :
    import cupy as cp
    GPU_exist = True
except ImportError :
    GPU_exist = False
    print("No GPU available")

print("GPU_exist", GPU_exist)

class TensorCoclusteringBernoulli(BaseNonDiagonalCoclust):
    """Tensor Latent Block Model for Bernoulli distribution.
    
    Parameters
    ----------
    n_row_clusters : int, optional, default: 2
        Number of row clusters to form
    n_col_clusters : int, optional, default: 2
        Number of column clusters to form
    fuzzy : boolean, optional, default: True
        Provide fuzzy clustering, If fuzzy is False
        a hard clustering is performed
    init_row : numpy array or scipy sparse matrix, \
        shape (n_rows, K), optional, default: None
        Initial row labels
    init_col : numpy array or scipy sparse matrix, \
        shape (n_cols, L), optional, default: None
        Initial column labels
    max_iter : int, optional, default: 20
        Maximum number of iterations
    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    tol : float, default: 1e-9
        Relative tolerance with regards to criterion to declare convergence
    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        Bicluster label of each row
    column_labels_ : array-like, shape (n_cols,)
        Bicluster label of each column
    mu_kl : array-like, shape (k,l,v)
        Value :math: mean vector for each row
        cluster k and column cluster l
    """

    def __init__(self, n_row_clusters=2, n_col_clusters=2, fuzzy=False, init_row=None, init_col=None,
                 max_iter=50, n_init=1, tol=1e-6, random_state=None, gpu=None):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.init_row = init_row
        self.init_col = init_col
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.fuzzy = fuzzy
        self.row_labels_ = None
        self.column_labels_ = None
        self.criterions = []
        self.criterion = -np.inf
        self.mu_kl = None
        self.mu_kl_evolution = None
        self.gpu = gpu

    def fit(self, X, y=None):
        """Perform Tensor co-clustering.
        
        Parameters
        ----------
        X : three-way numpy array, shape=(n_row_objects,d_col_objects, v_features)
            Tensor to be analyzed
        """
        global GPU_exist
        if self.gpu is None:
            self.gpu = GPU_exist
        else:
            GPU_exist = self.gpu

        random_state = check_random_state(self.random_state)

        # check_array(X, accept_sparse=True, dtype="numeric", order=None,
        #             copy=False, force_all_finite=True, ensure_2d=True,
        #             allow_nd=False, ensure_min_samples=self.n_row_clusters,
        #             ensure_min_features=self.n_col_clusters,
        #             warn_on_dtype=False, estimator=None)

        check_tensor(X)
        check_numbers_clusters_non_diago(X,self.n_row_clusters, self.n_col_clusters)

        X = X.astype(int)

        criterion = self.criterion
        criterions = self.criterions
        row_labels_ = self.row_labels_
        column_labels_ = self.column_labels_
        mu_kl = self.mu_kl
        mu_kl_evolution = self.mu_kl_evolution
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self._fit_single(X, seed, y)
            if np.isnan(self.criterion):
                raise ValueError("matrix may contain negative or "
                                 "unexpected NaN values")
            # remember attributes corresponding to the best criterion
            if (self.criterion > criterion):
                criterion = self.criterion
                criterions = self.criterions
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_
                mu_kl_ = self.mu_kl
                mu_kl_evolution = self.mu_kl_evolution

        # update attributes
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_
        self.mu_kl_ = mu_kl_
        self.mu_kl_evolution = mu_kl_evolution

        return self

    def mukl(self, x, z, w):
        """Compute the mean vector mu_kl per bloc.
        
        Parameters
        ----------
        X : three-way numpy array, shape=(n_row_objects,d_col_objects, v_features)
            Tensor to be analyzed
        z : numpy array, shape= (n_row_objects, K)
            matrix of row partition
        w : numpy array, shape(d_col_objects, L)
            matrix of column partition
        Returns
        -------
        mukl_mat
            three-way numpy array
        """
        n = z.shape[0]
        d = w.shape[0]
        K = z.shape[1]
        L = w.shape[1]
        v = x.shape[2]

        # x = x.reshape(n,v,d)

        indRSelec = np.arange(n)
        indCSelec = np.arange(d)
        """
        sum_z = np.sum(z, 0).reshape(K, 1)
        sum_w = np.sum(w, 0).reshape(1, L)
        nbr_element_class = sum_z.dot(sum_w)
        print("nbr_element_class ", nbr_element_class)
        """
        mukl_mat = np.zeros((K, L, v))
        for k in range(K):
            z_k = z[:, k].reshape(n, 1)
            for l in range(L):
                w_l = w[:, l].reshape(1, d)
                poids = z_k.dot(w_l)
                nbr_element_class = np.sum(poids)
                if not GPU_exist:
                    dup_S = poids.reshape(n,d,1)# np.repeat(poids[:, :, np.newaxis], v, axis=2)
                    x_poids = np.multiply(dup_S, x)
                    sum_kl = np.sum(x_poids, axis=(0, 1))
                else:
                    x_gpu = cp.asarray(x)
                    poids_gpu = cp.asarray(poids)
                    dup_S = poids_gpu.reshape(n,d,1) #cp.repeat(poids_gpu[:, :, np.newaxis], v, axis=2)
                    x_poids = cp.multiply(dup_S, x_gpu)
                    sum_kl = cp.sum(x_poids, axis=(0, 1))
                    sum_kl= cp.asnumpy(sum_kl)
                    cp.cuda.Stream.null.synchronize()

                mukl_mat[k][l] = (sum_kl / nbr_element_class) + 1.e-6#  (nbr_element_class[k, l] + 1.e-5)
        mukl_mat[mukl_mat>=1]=0.99
        return mukl_mat

    def pi_k(self,z):
        """Compute row proportion.
        
        Parameters
        ----------
        z : numpy array, shape= (n_row_objects, K)
            matrix of row partition
        Returns
        -------
        pi_k_vect
            numpy array, shape=(K)
            proportion of row clusters
        """
        n = z.shape[0]
        pi_k_vect = np.sum(z, 0) / n
        return pi_k_vect

    def rho_l(self,w):
        """Compute column proportion.
        
        Parameters
        ----------
        w : numpy array, shape(d_col_objects, L)
            matrix of column partition
        Returns
        -------
        rho_l_vect
            numpy array, shape=(L)
            proportion of column clusters
        """
        d = w.shape[0]
        rho_l_vect = np.sum(w, 0) / d
        return rho_l_vect

    def F_c(self, x, z, w, mukl, pi_k, rho_l, choice='ZW'):
        """Compute fuzzy log-likelihood (LL) criterion.
        
        Parameters
        ----------
        X : three-way numpy array, shape=(n_row_objects,d_col_objects, v_features)
            Tensor to be analyzed
        z : numpy array, shape= (n_row_objects, K)
            matrix of row partition
        w : numpy array, shape(d_col_objects, L)
            matrix of column partition
        mukl : three-way numpy array, shape=(K,L, v_features)
            matrix of mean parameter pe bloc
        pi_k : numpy array, shape(K,)
            vector of row cluster proportion
        rho_l : numpy array, shape(K,)
            vector of column cluster proportion
        choice : string, take values in ("Z", "W", "ZW")
            considering the optimization of LL
        Returns
        -------
        (H_z, H_w, LL, value)
            (row entropy, column entropy, Log-likelihood, lower bound of log-likelihood)
        """
        n = z.shape[0]
        d = w.shape[0]
        K = z.shape[1]
        L = w.shape[1]
        v = x.shape[2]  # Nombre de covariates
        # Reshape X matrix
        Xij_selec = x.reshape(n * d, v)
        H_w = 0
        H_z = 0
        z_weight = 0
        w_weight = 0

        one3D = np.ones((n, d, v))
        LL = 0
        cpt = 0
        for k in range(K):
            z_k = z[:, k].reshape(n, 1)
            for l in range(L):
                w_l = w[:, l].reshape(1, d)
                poids = z_k.dot(w_l)
                # print('poids', poids.shape)
                zkwl = poids.reshape(n * d, 1)

                mukl_select = (mukl[k][l]).reshape(1, v)
                # print('Ixij',Ixij.shape)
                Imukl = np.log(np.ones((1, v)) - mukl_select)
                # print("erreur_y",erreur_y.shape)
                ################
                if not GPU_exist:
                    xijLnmukl = (x[:, :, :] * np.log(mukl_select[0, :])).reshape(n, d, v)
                    # print('xijLnmukl',xijLnmukl.shape)
                    Ixij = (one3D - x[:, :, :]).reshape(n, d, v)
                    Ixij_Imukl = (Ixij[:, :, :] * (Imukl[0, :])).reshape(n, d, v)
                else:
                    x_gpu = cp.asarray(x)
                    one3D_gpu = cp.asarray(one3D)
                    mukl_select_gpu = cp.asarray(mukl_select)
                    Imukl_gpu = cp.asarray(Imukl)
                    xijLnmukl = (x_gpu[:, :, :] * cp.log(mukl_select_gpu[0, :])).reshape(n, d, v)

                    # print('xijLnmukl',xijLnmukl.shape)
                    Ixij = (one3D_gpu - x_gpu[:, :, :]).reshape(n, d, v)
                    Ixij_Imukl = (Ixij[:, :, :] * (Imukl_gpu[0, :])).reshape(n, d, v)
                    xijLnmukl  = cp.asnumpy(xijLnmukl)
                    Ixij_Imukl = cp.asnumpy(Ixij_Imukl)
                    cp.cuda.Stream.null.synchronize()
                # print("Imukl",Imukl.shape)
                #########

                # print('Ixij_Imukl',Ixij_Imukl.shape)
                # a * b[:, None]
                poids_t = (poids.T)
                error = poids[:, :, None] * (xijLnmukl + (Ixij_Imukl))
                # print('error', error.shape)
                LL = LL + np.sum(error)
                cpt = cpt + 1

                # LL  = LL + ((-1)*n*d*np.log(2*np.pi))
        value = 0
        if choice == "ZW":
            H_z = 0
            for i in range(n):
                for k in range(K):
                    H_z = H_z - (z[i, k] * np.log(z[i, k]))
            H_w = 0
            for j in range(d):
                for l in range(L):
                    H_w = H_w - (w[j, l] * np.log(w[j, l]))

            z_weight = 0
            for k in range(K):
                z_weight = z_weight + (np.sum(z[:, k]) * np.log(pi_k[k]))

            w_weight = 0
            for l in range(L):
                w_weight = w_weight + (np.sum(w[:, l]) * np.log(rho_l[l]))

            value = z_weight + w_weight + LL  # + H_z + H_w
        if choice == "Z":
            H_z = 0
            for i in range(n):
                for k in range(K):
                    H_z = H_z - (z[i, k] * np.log(z[i, k]))

            z_weight = 0
            for k in range(K):
                z_weight = z_weight + (np.sum(z[:, k]) * np.log(pi_k[k]))

            value = z_weight + LL + H_z
        if choice == "W":

            H_w = 0
            for j in range(d):
                for l in range(L):
                    H_w = H_w - (w[j, l] * np.log(w[j, l]))

            w_weight = 0
            for l in range(L):
                w_weight = w_weight + (np.sum(w[:, l]) * np.log(rho_l[l]))

            value = w_weight + LL + H_w

        return [H_z, H_w, LL, value]


    def _fit_single(self, data, random_state, y=None):
        """Perform one run of Tensor co-clustering.
        
        Parameters
        ----------
        X : three-way numpy array, shape=(n_row_objects,d_col_objects, v_features)
            Tensor to be analyzed
        """
        K = self.n_row_clusters
        L = self.n_col_clusters
        bool_fuzzy = self.fuzzy
        if self.init_row is None:
            z = random_init(K, data.shape[0], random_state)

        else:
            z = np.array(self.init_row, dtype=float)


        if self.init_col is None:

            w = random_init(L, data.shape[1], random_state)
        else:

            w = np.array(self.init_col, dtype=float)
        ########################################################

        n = data.shape[0]
        d = data.shape[1]
        nbr_covariates = data.shape[2]
        ########################################################
        mukl_hat = self.mukl(data, z, w) + 1.e-8
        print("les mukl_hat", mukl_hat)
        pi_k_hat = self.pi_k(z)
        print("proportion lignes", pi_k_hat)
        rho_l_hat = self.rho_l(w)
        print("proportion colonnes", rho_l_hat)
        result = self.F_c(data, z, w, mukl_hat, pi_k_hat, rho_l_hat, choice='ZW')
        fc = result[3]
        print("objective function", fc)
        ########################################################
        ########################################################
        #################################
        # Début de l'algorithme BLVEM
        #################################
        iteration_n = self.max_iter
        iteration_z = int(10)
        iteration_w = int(10)

        #################################
        dessiner_courbe_evol_mukl= np.zeros((K, L, iteration_n + 1))
        for k in range(K):
            for l in range(L):
                dessiner_courbe_evol_mukl[k, l, 0] = np.mean(mukl_hat[k, l, :])
                #################################
        ########################################################
        LL = []
        LL.append(fc)
        fc_previous = float(-np.inf)
        t = 0
        change = True

        while change and t < self.max_iter:
            print("iteration n: ", t)
            t_z = 0
            while t_z < iteration_z:
                print("iteration t_z :", t_z)
                # E-step :
                z = np.float64(np.zeros((n, K)))
                for i in range(n):
                    x_select = np.asarray(data[[i], :, :]).reshape(d, nbr_covariates)
                    for k in range(K):
                        compound = 0
                        for l in range(L):
                            w_l = w[:, l].reshape(d, 1)
                            ##############
                            mukl_select = (mukl_hat[k][l]).reshape(1, nbr_covariates)
                            xijLog = (x_select[:, :] * np.log(mukl_select[0, :])).reshape(d, nbr_covariates)
                            oneXij = np.ones((d, nbr_covariates)) - x_select[:, :]
                            logOneMukl = np.log(np.ones((1, nbr_covariates)) - mukl_select)
                            ###############
                            # print((logOneMukl[0,:] * oneXij ).shape)
                            value = xijLog + (logOneMukl[0, :] * oneXij)
                            compound = compound + np.sum((w_l * value))


                        z[i, k] = np.log(pi_k_hat[k]) + compound

                    if bool_fuzzy== True :
                         #print("soft")
                         z[i, :] = z[i, :] - np.amax(z[i, :])
                         z[i, :] = np.exp(z[i, :]) / np.sum(np.exp(z[i, :])) + 1.e-5
                    else:
                         #print("hard")
                         ind_max_r = np.argmax(z[i, :])
                         z[i, :] = 0 + 1.e-10
                         z[i, ind_max_r] = 1

                    # print("z", z)
                # M-step :
                pi_k_hat = self.pi_k(z)
                mukl_hat = self.mukl(data, z, w)

                # Calculer LL :

                t_z = t_z + 1

                ########################################################
            t_w = 0
            while t_w < iteration_w:
                print("iteration t_w :", t_w)
                # E-step :
                w = np.float64(np.zeros((d, L)))
                for j in range(d):
                    x_select = np.asarray(data[:, [j], :]).reshape(n, nbr_covariates)
                    for l in range(L):
                        compound = 0
                        for k in range(K):
                            z_k = z[:, k].reshape(n, 1)
                            ###############################
                            mukl_select = (mukl_hat[k][l]).reshape(1, nbr_covariates)
                            xijLog = (x_select[:, :] * np.log(mukl_select[0, :])).reshape(n, nbr_covariates)
                            oneXij = np.ones((n, nbr_covariates)) - x_select[:, :]
                            logOneMukl = np.log(np.ones((1, nbr_covariates)) - mukl_select)
                            ###############################
                            value = xijLog + (logOneMukl[0, :] * oneXij)
                            compound = compound + np.sum((z_k * value))

                        w[j, l] = np.log(rho_l_hat[l]) + compound

                    if bool_fuzzy == True:
                        #print("soft")
                        w[j, :] = w[j, :] - np.amax(w[j, :])
                        w[j, :] = np.exp(w[j, :]) / np.sum(np.exp(w[j, :])) + 1.e-5
                    else:
                        #print("hard")
                        ind_max_c = np.argmax(w[j,:] )
                        w[j,:]   = 0 + 1.e-10
                        w[j,ind_max_c] = 1
                    # M-step :

                rho_l_hat = self.rho_l(w)
                mukl_hat = self.mukl(data, z, w)

                # Calcul LL :

                t_w = t_w + 1

            for k in range(K):
                for l in range(K):
                    dessiner_courbe_evol_mukl[k, l, t + 1] = np.mean(mukl_hat[k, l, :])

            result = self.F_c(data, z, w, mukl_hat, pi_k_hat, rho_l_hat, choice='ZW')
            fc = result[3]
            LL.append(fc)
            print("fc value", fc)

            if np.abs(fc - fc_previous) > self.tol:
                fc_previous = fc
                change = True
                LL.append(fc)
                print("fc value", fc)
                t = t+1
            else :
                change = False


        ########################################################

        self.criterions = LL
        self.criterion = fc
        self.row_labels_ = np.argmax(z, 1) + 1
        self.column_labels_ = np.argmax(w, 1) + 1
        self.mu_kl = mukl_hat
        self.mu_kl_evolution = dessiner_courbe_evol_mukl
        self.Z = z
        self.W = w
