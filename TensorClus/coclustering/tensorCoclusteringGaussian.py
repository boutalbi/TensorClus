# -*- coding: utf-8 -*-

"""

The :mod:`TensorClus.coclustering.tensorCoclusteringGaussian` module provides an implementation
of a tensor co-clustering algorithm for continous three-way tensor.
"""

# Author: Rafika Boutalbi <rafika.boutalbi@gmail.com>
#         Mohamed Nadif <mohamed.nadif@u-paris.fr>
#         Lazhar Labiod <lazhar.labiod@u-paris.fr>

# License: BSD 3 clause

from __future__ import division
import numpy as np
from numpy import linalg

import random

from numpy.linalg import inv

from sklearn.utils import check_random_state

from ..initialization import random_init
from .baseNonDiagonalCoclustering import BaseNonDiagonalCoclust
from ..tests.input_checking import check_tensor, check_numbers_clusters_non_diago


GPU_exist = False
try :
    import cupy as cp
    GPU_exist = True
except ImportError :
    GPU_exist = False
    print("No GPU available")

print("GPU_exist", GPU_exist)

class TensorCoclusteringGaussian(BaseNonDiagonalCoclust):
    """Tensor Latent Block Model for Normal distribution.
    
    Parameters
    ----------    
    n_row_clusters : int, optional, default: 2
        Number of row clusters to form
    n_col_clusters : int, optional, default: 2
        Number of column clusters to form
    fuzzy : boolean, optional, default: True
        Provide fuzzy clustering, If fuzzy is False
        a hard clustering is performed
    parsimonious : boolean, optional, default: True
        Provide parsimonious model, If parsimonious False
        sigma is computed at each iteration
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
    sigma_kl_ : array-like, shape (k,l,v,v)
        Value of covariance matrix for each row
        cluster k and column cluster
    """


    def __init__(self, n_row_clusters=2, n_col_clusters=2, fuzzy=True, parsimonious= True, init_row=None, init_col=None,
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
        self.parsimonious = parsimonious
        self.row_labels_ = None
        self.column_labels_ = None
        self.criterions = []
        self.criterion = -np.inf
        self.mu_kl = None
        self.mu_kl_evolution = None
        self.sigma_kl_= None
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

        criterion = self.criterion
        criterions = self.criterions
        row_labels_ = self.row_labels_
        column_labels_ = self.column_labels_
        mu_kl = self.mu_kl
        mu_kl_evolution = self.mu_kl_evolution
        sigma_kl_ = self.sigma_kl_

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        #seeds = random.sample(range(10, 30), self.n_init)
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
                mu_kl = self.mu_kl
                mu_kl_evolution = self.mu_kl_evolution
                sigma_kl_ = self.sigma_kl_

        # update attributes
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_
        self.mu_kl = mu_kl
        self.mu_kl_evolution = mu_kl_evolution
        self.sigma_kl_ = sigma_kl_

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
            three-way numpy array, shape=(K,L, v_features)
            Computed parameters per block
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

        return mukl_mat

    def sigma_x_kl(self, x, z, w, mukl):
        """Compute the mean vector sigma_kl per bloc.

        Parameters
        ----------
        X : three-way numpy array, shape=(n_row_objects,d_col_objects, v_features)
            Tensor to be analyzed
        z : numpy array, shape= (n_row_objects, K)
            matrix of row partition
        w : numpy array, shape(d_col_objects, L)
            matrix of column partition
        mukl : numpy array, shape(K,L, v_features)
            tensor of mukl values
        Returns
        -------
        sigma_x_kl_mat
            three-way numpy array
            Computed the covariance parameters per block
        """
        n = z.shape[0]
        d = w.shape[0]
        K = z.shape[1]
        L = w.shape[1]
        v = x.shape[2]  # Nombre de covariates

        sum_z = np.sum(z, 0).reshape(K, 1)
        sum_w = np.sum(w, 0).reshape(1, L)
        nbr_element_class = sum_z.dot(sum_w)

        # Reshape X matrix
        indRSelec = np.arange(n)
        indCSelec = np.arange(d)
        Xij_selec = x[indRSelec, :, :]
        Xij_selec = np.asarray(Xij_selec[:, indCSelec, :]).reshape(n * d, v)

        sigma_x_kl_mat = np.zeros((v, v, K * L))

        cpt = 0
        for k in range(K):
            z_k = z[:, k].reshape(n, 1)
            for l in range(L):
                if self.parsimonious == False :

                    w_l = w[:, l].reshape(1, d)
                    poids = z_k.dot(w_l)
                    zkwl = poids.reshape(n * d)
                    mukl_act = (mukl[k][l]).reshape(1, v)
                    euclDist = (Xij_selec[:, :] - mukl_act[0, :]).reshape(n * d, v)
                    euclDist_T = euclDist.T
                    euclDist_T_S = euclDist_T * zkwl
                    error = euclDist_T_S.dot(euclDist)
                    sigma_x_kl_mat[:, :, cpt] = error / (np.sum(zkwl) + 1.e-5)
                    # print(sigma_x_kl_mat[:,:,cpt]
                else:
                    sigma_x_kl_mat[:, :, cpt] = np.identity(v)

                cpt = cpt + 1

        return sigma_x_kl_mat

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
        pi_k_vect = np.sum(z, 0) / n + 1.e-9
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
        rho_l_vect = np.sum(w, 0) / d + 1.e-9
        return rho_l_vect

    def F_c(self, x, z, w, mukl, sigma_x_kl, pi_k, rho_l, choice='ZW'):
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
        sigma_x_kl : Four-way numpy array, shape=(K,L,v_features, v_features)
           tensor of sigma matrices for all blocks
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
                # print("erreur_y",erreur_y.shape)
                ################
                euclDist = (x[:, :, :] - mukl_select[0, :]).reshape(n, d, v)
                # print('euclDist',euclDist.shape)
                sigma_x_act = (sigma_x_kl[:, :, cpt]).reshape(v, v)
                inv_sigma_x = inv(sigma_x_act)
                # S = np.repeat(poids[:, :, np.newaxis], v-1, axis=2)
                # euclDist_S = np.multiply(euclDist,S)
                # print("euclDist_S",euclDist_S.shape)
                euclDist_T = euclDist.T
                if not GPU_exist:
                    euclDist_inv = euclDist.dot(inv_sigma_x)
                    erreur_x = np.einsum('abi,jba->ab', euclDist_inv, euclDist_T)
                    # print("erreur_x",erreur_x.shape)
                else:
                    inv_sigma_x_gpu = cp.asarray(inv_sigma_x)
                    euclDist_gpu = cp.asarray(euclDist)
                    euclDist_T_gpu  = cp.asarray(euclDist_T)
                    euclDist_inv = euclDist_gpu.dot(inv_sigma_x_gpu)
                    erreur_x = cp.einsum('abi,jba->ab', euclDist_inv, euclDist_T_gpu)
                    erreur_x = cp.asnumpy(erreur_x)
                    cp.cuda.Stream.null.synchronize()
                #########
                # print('SigmaY_Sigma_X',SigmaY_Sigma_X.shape)
                error = (poids * (-1 / 2) * (np.log(np.linalg.det(sigma_x_act)) + erreur_x))
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
        mukl_hat = self.mukl(data, z, w)
        print("les mukl_hat", mukl_hat)
        sigma_x_kl_hat = self.sigma_x_kl(data, z, w, mukl_hat)
        print("les sigma_x_kl_hat", sigma_x_kl_hat)
        pi_k_hat = self.pi_k(z)
        print("proportion lignes", pi_k_hat)
        rho_l_hat = self.rho_l(w)
        print("proportion colonnes", rho_l_hat)
        result = self.F_c(data, z, w, mukl_hat,sigma_x_kl_hat, pi_k_hat, rho_l_hat, choice='ZW')
        fc = result[3]
        print("objective function", fc)
        ########################################################
        ########################################################
        #################################
        # Début de l'algorithme BLVEM
        #################################
        iteration_n = self.max_iter
        iteration_z = int(1)
        iteration_w = int(1)
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
                z = np.float32(np.zeros((n, K)))
                for i in range(n):
                    x_select = np.asarray(data[[i], :, :]).reshape(d, nbr_covariates)
                    cpt1 = 0
                    for k in range(K):
                        compound = 0
                        for l in range(L):
                            w_l = (-1 / 2) * w[:, l].reshape(d, 1)
                            ##############
                            mukl_select = (mukl_hat[k][l]).reshape(1, nbr_covariates)
                            sigma_x_act = (sigma_x_kl_hat[:, :, cpt1]).reshape(nbr_covariates, nbr_covariates)
                            inv_sigma_x = inv(sigma_x_act)
                            euclidDist = (x_select[:, :] - mukl_select[0, :]).reshape(d, nbr_covariates)
                            euclDist_T = euclidDist.T
                            euclDist_inv = euclidDist.dot(inv_sigma_x)
                            euclDist_inv_euclDist_T = euclDist_inv.dot(euclDist_T)
                            erreur_x = (np.diag(euclDist_inv_euclDist_T)).reshape(d, 1)
                            ###############
                            value = (np.log(2 * np.pi * np.linalg.det(sigma_x_act)) + erreur_x).reshape(d, 1)
                            compound = compound + np.sum((w_l * value))
                            cpt1 = cpt1 + 1

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
                sigma_x_kl_hat = self.sigma_x_kl(data, z, w, mukl_hat)

                # Calculer LL :

                t_z = t_z + 1

                ########################################################
            t_w = 0
            while t_w < iteration_w:
                print("iteration t_w :", t_w)
                # E-step :
                w = np.float32(np.zeros((d, L)))
                for j in range(d):
                    x_select = np.asarray(data[:, [j], :]).reshape(n, nbr_covariates)
                    for l in range(L):
                        compound = 0
                        cpt2 = int(l)
                        for k in range(K):
                            z_k = (-1 / 2) * z[:, k].reshape(n, 1)

                            ###############################
                            mukl_select = (mukl_hat[k][l]).reshape(1, nbr_covariates)
                            sigma_x_act = (sigma_x_kl_hat[:, :, cpt2]).reshape(nbr_covariates, nbr_covariates)
                            inv_sigma_x = inv(sigma_x_act)
                            euclidDist = (x_select[:, :] - mukl_select[0, :]).reshape(n, nbr_covariates)
                            euclDist_T = euclidDist.T
                            euclDist_inv = euclidDist.dot(inv_sigma_x)
                            euclDist_inv_euclDist_T = euclDist_inv.dot(euclDist_T)
                            ###############################
                            erreur_x = (np.diag(euclDist_inv_euclDist_T)).reshape(n, 1)
                            value = (np.log(2 * np.pi * np.linalg.det(sigma_x_act)) + erreur_x).reshape(n, 1)
                            compound = compound + np.sum((z_k * value))
                            cpt2 = cpt2 + L
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
                sigma_x_kl_hat = self.sigma_x_kl(data, z, w, mukl_hat)
                # Calcul LL :

                t_w = t_w + 1

            for k in range(K):
                for l in range(K):
                    dessiner_courbe_evol_mukl[k, l, t + 1] = np.mean(mukl_hat[k, l, :])

            result = self.F_c(data, z, w, mukl_hat, sigma_x_kl_hat, pi_k_hat, rho_l_hat, choice='ZW')
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
        self.sigma_kl_ = sigma_x_kl_hat
        self.Z = z
        self.W = w
