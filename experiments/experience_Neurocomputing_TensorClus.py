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


from  tensorly.decomposition import parafac
from  tensorly.decomposition import tucker


import numpy as np
import pandas as pd 
import time


from coclust.evaluation.external import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

##################################################################
#                        Load DBLP1 dataset                      #
##################################################################
data_v2, labels, slices = load.load_dataset("DBLP1_dataset")

##################################################################
#                           Experiments                          #
##################################################################
decompositionNames   =["Tucker_Decomp", "Parafac"]
listAlgorithmes = ["Kmeans++", "Skmeans","SpectralClustering","GMM"]
listRankNumber  = [10,50,100]
nbrIteration = 30
df_results   = pd.DataFrame(columns=["algorithm", "n_rank", "ACC", "NMI", "ARI", "Time"], index = np.arange(nbrIteration*((len(decompositionNames)*len(listAlgorithmes)*len(listRankNumber))+1)).tolist())
print("df_results", df_results)
cpt= 0 
for t in range(nbrIteration):
    print("iteration ", t)
    ##################################################################
    #              Execute TSPLBM on the dataset                     #
    ##################################################################
    n = data_v2.shape[0]
    d = data_v2.shape[1]
    v = data_v2.shape[2]
    
    # Define the number of clusters K 
    K = 3
    
    # Optional: initialization of rows and columns partitions
    z=np.zeros((n,K))
    z_a=np.random.randint(K,size=n)
    z=np.zeros((n,K))+ 1.e-9
    z[np.arange(n) , z_a]=1 
                    
    w=np.asarray(z)
    
    ##################################################################    
    # Run TSPLBM  
    ##################################################################
    algoName = "TSPLBM"
    start_time = time.time()
    model = tcSCoP.SparseTensorCoclusteringPoisson(n_clusters=K ,  fuzzy = True,init_row=z, init_col=w,max_iter=50)
    model.fit(data_v2)
    end_time = time.time()
    timeExecution = end_time -start_time
    
    predicted_row_labels = model.row_labels_
    predicted_column_labels = model.column_labels_
    
    acc = np.around(accuracy(labels, predicted_row_labels),3)
    nmi = np.around(normalized_mutual_info_score(labels, predicted_row_labels),3)
    ari = np.around(adjusted_rand_score(labels, predicted_row_labels),3)
    print("Accuracy : ", acc)
    print("nmi : ", nmi)
    print("ari : ", ari)
    
    df_results.algorithm[cpt]= algoName
    print("df_results", df_results)
    df_results.n_rank[cpt]= str(0)
    df_results.ACC[cpt]=  str(acc)
    df_results.NMI[cpt]=  str(nmi)
    df_results.ARI[cpt]=  str(ari)
    df_results.Time[cpt]=  str(timeExecution)
    
    cpt = cpt + 1
    ##################################################################
    #                   Decomposition
    ##################################################################
    for r,rank in enumerate(listRankNumber):
        print("rank ", rank)
        # Test Tucker 
        start_time = time.time()
        factor , res_tucker= tucker(data_v2, rank,init='random', tol=10e-6)
        del factor
        end_time = time.time()
        timeExcutionTucker =  end_time -start_time
        ##################################################################
        # Test Parafac
        start_time = time.time()
        error , res_Parafac= parafac(data_v2, rank,init='random', tol=10e-6)
        del error
        end_time = time.time()
        timeExcutionParafac =  end_time -start_time    
        decompositionResults = [res_tucker, res_Parafac]
        timeExecutionResults = [timeExcutionTucker, timeExcutionParafac]
        ##################################################################
        # Test decomposition with clustering 
        for d,decomp in enumerate(decompositionResults):
            print("decomp ", decomp)
            for a,algo in enumerate(listAlgorithmes):
                print("algo ", algo)
                
                algoName = decompositionNames[d]+" + "+ listAlgorithmes[a]
                start_time = time.time()
                model_decompClustering = decomposition.DecompositionWithClustering(n_clusters = [3], modes = [1], algorithm = algo)
                model_decompClustering.fit(decomp)
                end_time = time.time()
                timeExecution = (end_time -start_time) + timeExecutionResults[d]
                
                all_clustering = model_decompClustering.labels_
            
                mode1_clustering = all_clustering[0] 
                print("mode1_clustering ", mode1_clustering)
                acc = np.around(accuracy(labels, mode1_clustering),3)
                nmi = np.around(normalized_mutual_info_score(labels, mode1_clustering),3)
                ari = np.around(adjusted_rand_score(labels, mode1_clustering),3)
                print("Accuracy decompostion with "+ algo + " : ", acc)
                
                df_results.algorithm[cpt]= algoName
                df_results.n_rank[cpt]=  str(rank)
                df_results.ACC[cpt]=  str(acc)
                df_results.NMI[cpt]=  str(nmi)
                df_results.ARI[cpt]=  str(ari)
                df_results.Time[cpt]=  str(timeExecution)
                
                cpt = cpt + 1


print("df_results", df_results)
df_results.to_csv("results_JMLR.csv", index=False)
