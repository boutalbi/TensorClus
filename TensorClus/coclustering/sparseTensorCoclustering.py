# -*- coding: utf-8 -*-

"""

The :mod:`TensorClus.coclustering.sparseTensorCoclustering` module provides an implementation
of a Sparse tensor co-clustering algorithm.
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
from .baseDiagonalCoclustering import BaseDiagonalCoclust
from TensorClus.tests.input_checking import check_tensor, check_numbers_clusters
# Test GPU availability

GPU_exist = False
try :
    import cupy as cp
    GPU_exist = True
except ImportError :
    GPU_exist = False
    print("No GPU available")

print("GPU_exist", GPU_exist)

class SparseTensorCoclusteringPoisson(BaseDiagonalCoclust):
    """Tensor Latent Block Model for Poisson distribution.

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
    gamma_kl : array-like, shape (k,l,v)
        Value :math:`\\frac{p_{kl}}{p_{k.} \\times p_{.l}}` for each row
        cluster k and column cluster l
    gamma_kl_evolution : array-like, shape(k,l,max_iter)
        Value of gamma_kl of each bicluster according to iterations
    """

    def __init__(self, n_clusters=2,  fuzzy = True, init_row=None, init_col=None,
                 max_iter=50, n_init=1, tol=1e-6, random_state=None, gpu = None):
        self.n_clusters = n_clusters
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
        self.gamma_kl = None
        self.gamma_kl_evolution = None
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
        #             copy=False, force_all_finite=True, ensure_2d=False,
        #             allow_nd=False, ensure_min_samples=self.n_row_clusters,
        #             ensure_min_features=self.n_col_clusters,
        #             warn_on_dtype=False, estimator=None)


        check_tensor(X)
        check_numbers_clusters(X,self.n_clusters)

        X = X.astype(int)

        criterion = self.criterion
        criterions = self.criterions
        row_labels_ = self.row_labels_
        column_labels_ = self.column_labels_
        gamma_kl = self.gamma_kl
        gamma_kl_evolution = self.gamma_kl_evolution
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
                gamma_kl = self.gamma_kl
                gamma_kl_evolution = self.gamma_kl_evolution
        # update attributes
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_
        self.gamma_kl = gamma_kl
        self.gamma_kl_evolution = gamma_kl_evolution

        return self

    def gammakl(self, x, z, w):
        """Perform Tensor co-clustering.

        Parameters
        ----------
        x : three-way numpy array, shape=(n_row_objects,d_col_objects, v_features)
            Tensor to be analyzed
        z : row partition
        w : column partition
        Returns
        -------
        gamma_kl_mat
            three-way numpy array, shape=(K,L, v_features)
            Computed parameters per block
        """
        n = z.shape[0]
        d = w.shape[0]
        K = z.shape[1]
        v = x.shape[2]


        if not GPU_exist :
            const = x.sum(axis=(0, 1)).reshape(1, v).astype(np.float64)
        else:
            x_gpu = cp.asarray(x)
            const = x_gpu.sum(axis=(0, 1)).reshape(1, v).astype(np.float64)
            const = cp.asnumpy(const)
            cp.cuda.Stream.null.synchronize()

        sum_z = np.sum(z, 0).reshape(K, 1)
        sum_w = np.sum(w, 0).reshape(1, K)
        nbr_element_class = sum_z.dot(sum_w)

        if not GPU_exist :
            xi = x.sum(axis=1)
            xj = x.sum(axis=0)
        else:
            xi = x_gpu.sum(axis=1)
            xi = cp.asnumpy(xi)
            xj = x_gpu.sum(axis=0)
            xj = cp.asnumpy(xj)
            cp.cuda.Stream.null.synchronize()

        gamma_kl_mat = np.zeros((K, K, v))
        zxw_mat = np.zeros((K, K, v))
        xkxl_mat = np.zeros((K, K, v))
        for k in range(K):
            z_k = z[:, k].reshape(n, 1)
            z_k_t = z_k.T
            z_k_t_xi = z_k_t.dot(xi)
            # print('z_k_t_xi', z_k_t_xi.shape)
            for l in range(K):
                if k == l:
                    w_l = w[:, l].reshape(1, d)
                    w_l_xj = w_l.dot(xj)

                    xkxl = z_k_t_xi * w_l_xj
                    xkxl_mat[k][l] = xkxl.reshape(v).tolist()
                    # print('xkxl.shape', xkxl.shape)
                    if not GPU_exist:
                        zx = np.einsum('ijk,il', x, z_k)
                    else:
                        zx = cp.einsum('ijk,il', x_gpu, z_k)
                        zx = cp.asnumpy(zx)
                        cp.cuda.Stream.null.synchronize()

                    zx = zx.reshape(d, v)
                    # print(zx.shape)
                    zxw = w_l.dot(zx)
                    zxw_mat[k][l] = zxw.reshape(v).tolist()
                    # print('zxw', zxw.shape)

                    value = (zxw / xkxl).reshape(v)

                    # print('value.shape', value.shape)
                    gamma_kl_mat[k][l] = (value).tolist()

        ####################################################################
        diag_zxw_mat = zxw_mat.diagonal(0, 0, 1)  # the "middle" (row) axis first.
        # print(diag_zxw_mat.shape)
        sum_zxw = np.sum(diag_zxw_mat, axis=1).reshape(1, v)
        diag_xkxl_mat = xkxl_mat.diagonal(0, 0, 1)
        sum_xkxl = np.sum(diag_xkxl_mat, axis=1).reshape(1, v)
        part1 = const - sum_zxw + 1.e-9
        part2 = (const * const) - sum_xkxl + 1.e-9
        vect_gamma = (part1 / part2).reshape(v).tolist()
        for k in range(K):
            for l in range(K):
                if k != l:
                    gamma_kl_mat[k][l] = vect_gamma


        return gamma_kl_mat  # + 1.e-9


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

    def F_c(self, x, z, w, gammakl, pi_k, rho_l, choice='ZW'):
        """Compute fuzzy log-likelihood (LL) criterion.

        Parameters
        ----------
        X : three-way numpy array, shape=(n_row_objects,d_col_objects, v_features)
            Tensor to be analyzed
        z : numpy array, shape= (n_row_objects, K)
            matrix of row partition
        w : numpy array, shape(d_col_objects, L)
            matrix of column partition
        gammakl : three-way numpy array, shape=(K,L, v_features)
            matrix of bloc's parameters
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
        # Xij_selec = x.reshape(n*d,v)

        if not GPU_exist :
            const = x.sum(axis=(0, 1)).reshape(1, v)
        else:
            x_gpu = cp.asarray(x)
            const = x_gpu.sum(axis=(0, 1)).reshape(1, v).astype(np.float64)
            const = cp.asnumpy(const)
            cp.cuda.Stream.null.synchronize()

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

        if not GPU_exist :
            xi = x.sum(axis=1)
            xj = x.sum(axis=0)
        else:
            xi = x_gpu.sum(axis=1)
            xi = cp.asnumpy(xi)
            xj = x_gpu.sum(axis=0)
            xj = cp.asnumpy(xj)
            cp.cuda.Stream.null.synchronize()

        LL = 0
        cpt = 0
        gamma_all = (gammakl[0][1]).reshape(1, v)

        for k in range(K):
            z_k = z[:, k].reshape(n, 1)
            z_k_t = z_k.T
            z_k_t_xi = z_k_t.dot(xi)

            w_l = w[:, k].reshape(1, d)
            w_l_xj = w_l.dot(xj)

            gkl = (gammakl[k][k]).reshape(1, v)

            div_gkl_gamma_all = (gkl / gamma_all)   + 1.e-9

            log_gkl = np.log(div_gkl_gamma_all).reshape(1, v)

            xkxl = z_k_t_xi * w_l_xj
            xkxl_gamma = xkxl * (gkl - gamma_all)
            # print('xkxl.shape', xkxl.shape)
            if not GPU_exist:
                zx = np.einsum('ijk,il', x, z_k)
            else:
                zx = cp.einsum('ijk,il', x_gpu, z_k)
                zx = cp.asnumpy(zx)
                cp.cuda.Stream.null.synchronize()


            zx = zx.reshape(d, v)

            zxw = w_l.dot(zx)
            zxw_gamma = zxw * log_gkl

            N_log_gamma = const / K * (np.log(gamma_all) - gamma_all * const).reshape(1, v)

            den = zxw_gamma - xkxl_gamma + N_log_gamma
            # print((den*gkl).shape)
            LL = LL + np.sum(den)

            cpt = cpt + 1
            # LL  = LL + ((-1)*n*d*np.log(2*np.pi))


        value = 0
        if choice == "ZW":
            value = z_weight + w_weight + LL  # + H_z + H_w
        if choice == "Z":
            value = z_weight + LL + H_z
        if choice == "W":
            value = w_weight + LL + H_w

        return [H_z, H_w, LL, value]

    def _fit_single(self, data, random_state, y=None):
        """Perform one run of Tensor co-clustering.

        Parameters
        ----------
        X : three-way numpy array, shape=(n_row_objects,d_col_objects, v_features)
            Tensor to be analyzed
        """
        K = self.n_clusters

        bool_fuzzy = self.fuzzy
        if self.init_row is None:
            z = random_init(K, X.shape[0], random_state)

        else:
            z = np.array(self.init_row, dtype=float)


        if self.init_col is None:

            w = random_init(K, X.shape[1], random_state)
        else:

            w = np.array(self.init_col, dtype=float)
        ########################################################


        n = data.shape[0]
        d = data.shape[1]
        nbr_covariates = data.shape[2]

        if not GPU_exist :
            const = 1. / (1. * data.sum(axis=(0, 1)) ** 2)
        else:
            data_gpu = cp.asarray(data)
            const = 1. / (1. * data_gpu.sum(axis=(0, 1)) ** 2)
            const = cp.asnumpy(const)
            cp.cuda.Stream.null.synchronize()
        ########################################################
        gammakl_hat = self.gammakl(data, z, w)
        print("les gammakl_hat", gammakl_hat)
        pi_k_hat = self.pi_k(z)
        print("proportion lignes", pi_k_hat)
        rho_l_hat = self.rho_l(w)
        print("proportion colonnes", rho_l_hat)
        result = self.F_c(data, z, w, gammakl_hat, pi_k_hat, rho_l_hat, choice='ZW')
        fc = result[3]
        print("objective function", fc)
        ########################################################
        iteration_n = self.max_iter
        iteration_z = int(10)
        iteration_w = int(10)


        #################################
        dessiner_courbe_evol_gammaKK = np.zeros((K, K, iteration_n + 1))
        for k in range(K):
            for l in range(K):
                dessiner_courbe_evol_gammaKK[k, l, 0] = np.mean(gammakl_hat[k, l, :])
                #################################
        ########################################################
        #################################
        # Début de l'algorithme BLVEM
        #################################
        LL = []
        LL.append(fc)
        fc_previous = float(-np.inf)
        t = 0
        change = True
        if not GPU_exist :
            xi = data.sum(axis=1)
            xj = data.sum(axis=0)
        else:
            xi = data_gpu.sum(axis=1)
            xi = cp.asnumpy(xi)
            xj = data_gpu.sum(axis=0)
            xj = cp.asnumpy(xj)
            cp.cuda.Stream.null.synchronize()

        while change and t < iteration_n:
            print("iteration n: ", t)
            t_z = 0
            while t_z < iteration_z:
                #print("iteration t_z :", t_z)
                # E-step :


                z = np.float64(np.zeros((n, K))) + 1.e-9
                gamma_all = (gammakl_hat[0][1]).reshape(1, nbr_covariates)
                for i in range(n):
                    xi_1 = xi[i, :].reshape(1, nbr_covariates)
                    xij = (data[i, :, :]).reshape(d, nbr_covariates)
                    for k in range(K):
                        pik = pi_k_hat[k]
                        gammakl_values = (gammakl_hat[k][k]).reshape(1, nbr_covariates)
                        div_gkl_gamma_all = gammakl_values / gamma_all
                        loggammakl = np.log(div_gkl_gamma_all)
                        part1 = xi_1 * xj
                        # print('part1', part1.shape)
                        part2 = (gammakl_values - gamma_all) * -part1
                        # print('part2', part2.shape)
                        part3 = xij * loggammakl
                        sum_jl = 0
                        for l in range(K):
                            if l == k:
                                wjl = w[:, l].reshape(d, 1)
                                # print('part3', part3.shape)
                                part4 = wjl * (part2 + part3)  #
                                # print('part4', part4.shape)
                                sum_jl = sum_jl + np.sum(part4)

                        z[i, k] = np.log(pik) + sum_jl

                    # ind_max_r = np.argmax(z[i,:] )
                    # z[i,:]  = 0 + 1.e-10
                    # z[i,ind_max_r] = 1
                    if bool_fuzzy== True :
                         #print("soft")
                         z[i, :] = z[i, :] - np.amax(z[i, :])
                         z[i, :] = np.exp(z[i, :]) / np.sum(np.exp(z[i, :])) + 1.e-5
                    else:
                         #print("hard")
                         ind_max_r = np.argmax(z[i, :])
                         z[i, :] = 0 + 1.e-10
                         z[i, ind_max_r] = 1
                    # print(z)
                # M-step :
                pi_k_hat = self.pi_k(z)
                gammakl_hat = self.gammakl(data, z, w)

                # Calculer LL :

                t_z = t_z + 1

            t_w = 0
            while t_w < iteration_w:
                #print("iteration t_w :", t_w)
                # E-step :
                w = np.float64(np.zeros((d, K))) + 1.e-9

                gamma_all = (gammakl_hat[0][1]).reshape(1, nbr_covariates)
                for j in range(d):
                    xj_1 = xj[j, :].reshape(1, nbr_covariates)
                    xij = (data[:, j, :]).reshape(n, nbr_covariates)
                    for l in range(K):
                        rohl = rho_l_hat[l]
                        gammakl_values = (gammakl_hat[l][l]).reshape(1, nbr_covariates)
                        div_gkl_gamma_all = gammakl_values / gamma_all
                        loggammakl = np.log(div_gkl_gamma_all)
                        part1 = xj_1 * xi
                        # print('part1', part1.shape)
                        part2 = (gammakl_values - gamma_all) * -part1
                        # print('part2', part2.shape)
                        part3 = xij * loggammakl
                        sum_ik = 0
                        for k in range(K):
                            if k == l:
                                zik = z[:, k].reshape(n, 1)

                                # print('part3', part3.shape)
                                part4 = zik * (part2 + part3)  #
                                # print('part4', part4.shape)
                                sum_ik = sum_ik + np.sum(part4)

                        w[j, l] = np.log(rohl) + sum_ik


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
                gammakl_hat = self.gammakl(data, z, w)

                # Calcul LL :

                t_w = t_w + 1

            for k in range(K):
                for l in range(K):
                    dessiner_courbe_evol_gammaKK[k, l, t + 1] = np.mean(gammakl_hat[k, l, :])

            result = self.F_c(data, z, w, gammakl_hat, pi_k_hat, rho_l_hat, choice='ZW')
            fc = result[3]
            LL.append(fc)
            print("fc value", fc)

            if np.abs(fc - fc_previous) > self.tol:
                fc_previous = fc
                change = True
                LL.append(fc)
                t = t+1
            else :
                change = False

        t_arret = int(t)
        dessiner_courbe_evol_gammaKK = dessiner_courbe_evol_gammaKK[:, :, 0:(t_arret + 1)]

        ########################################################

        self.criterions = LL
        self.criterion = fc
        self.row_labels_ = np.argmax(z, 1) + 1
        self.column_labels_ = np.argmax(w, 1) + 1
        self.gamma_kl = gammakl_hat
        self.gamma_kl_evolution = dessiner_courbe_evol_gammaKK
        self.Z = z
        self.W = w
