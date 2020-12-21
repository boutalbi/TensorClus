#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Rafika Boutalbi 21/07/2020


import TensorClus.coclustering.tensorCoclusteringPoisson as tcCoP
import TensorClus.coclustering.tensorCoclusteringBernoulli as tcCoB
import TensorClus.coclustering.tensorCoclusteringGaussian as tcCoG
import TensorClus.coclustering.sparseTensorCoclustering as tcSCoP
from TensorClus.vizualisation import plot_logLikelihood_evolution
from TensorClus.vizualisation import plot_parameter_evolution
from TensorClus.vizualisation import plot_slice_reorganisation
from TensorClus.reader import load
import TensorClus.decomposition.decomposition_with_clustering as decomposition

from tensorly.decomposition import parafac
from tensorly.decomposition import tucker

from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.cp import CP_ALS
import tensorD.demo.DataGenerator as dg
from tensorD.factorization.tucker import HOOI
from tensorD.factorization.ncp import NCP_BCU
from tensorD.factorization.ntucker import NTUCKER_BCU

import numpy as np
import pandas as pd

from coclust.evaluation.external import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

import tensorflow as tf
import time

bdd_name = ['DBLP2_dataset', 'PubMed Diabets_dataset', 'Nus_Wide_8_dataset']
K_lists  = [3,3,4]
df_time_execution = pd.DataFrame(columns=["Bdd_name", "Type", "Time"])
for i,name in enumerate(bdd_name):
    ##################################################################
    #                           Load dataset                         #
    ##################################################################
    data_v2, labels, slices = load.load_dataset(name)
    data_v2_float = data_v2.astype(np.float16)
    ##################################################################
    #              Execute TSPLBM on the dataset                     #
    ##################################################################
    n = data_v2.shape[0]
    d = data_v2.shape[1]
    v = data_v2.shape[2]

    # Define the number of clusters K
    K = K_lists[i]
    for it in range(10):
        # Optional: initialization of rows and columns partitions
        z = np.zeros((n, K))
        z_a = np.random.randint(K, size=n)
        z = np.zeros((n, K)) + 1.e-9
        z[np.arange(n), z_a] = 1

        w = np.asarray(z)

        type_boolean = [False, True]
        type_name = ["CPU", "GPU"]
        for j,type in enumerate(type_name):
            # Run TSPLBM
            s = time.time()
            model = tcSCoP.SparseTensorCoclusteringPoisson(n_clusters=K, fuzzy=True, init_row=z, init_col=w, max_iter=30, gpu=type_boolean[j])
            model.fit(data_v2_float)
            e = time.time()
            time_execution = e - s
            df_time_execution = df_time_execution.append({'Bdd_name': name, 'Type': type, 'Time': time_execution}, ignore_index=True)
            print("time_execution ", time_execution)

df_time_execution.to_csv("result_comparison_time.csv",index=False)
