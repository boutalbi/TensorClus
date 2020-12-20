#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Rafika Boutalbi 21/07/2020


import TensorClus.coclustering.sparseTensorCoclustering as tcSCoP
import TensorClus.decomposition.decomposition_with_clustering as decomposition
import numpy as np
from TensorClus.reader import load
from TensorClus.vizualisation import plot_parameter_evolution
from TensorClus.vizualisation import plot_logLikelihood_evolution
from TensorClus.vizualisation import plot_slice_reorganisation
from coclust.evaluation.external import accuracy
from tensorD.dataproc.provider import Provider
from tensorD.factorization.env import Environment
from tensorD.factorization.ntucker import NTUCKER_BCU
from tensorly.decomposition import tucker

##################################################################
#                Load dataset from txt file                      #
##################################################################
data_txt = load.read_txt_tensor("tensor_test.csv")
print("data_txt ", data_txt.shape)
load.save_txt_tensor(data_txt, "tensor_txt.csv")

##################################################################
#                        Load DBLP1 dataset                      #
##################################################################
data_v2, labels, slices = load.load_dataset("DBLP1_dataset")

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


# Run TSPLBM  

model = tcSCoP.SparseTensorCoclusteringPoisson(n_clusters=K ,  fuzzy = True,init_row=z, init_col=w,max_iter=10)
model.fit(data_v2)


predicted_row_labels = model.row_labels_
predicted_column_labels = model.column_labels_

acc = np.around(accuracy(labels, predicted_row_labels),3)
print("Accuracy : ", acc)

##################################################################
#                Factorization and clustering                    #
##################################################################
# Test Tucker using Tensorly
#error , res_Parafac= parafac(data_v2, 10,init='random', tol=10e-6)
factor , res_tucker= tucker(data_v2, 10,init='random', tol=10e-6)

##################################################################
# Test Tucker using TensorD

data_provider = Provider()
data_provider.full_tensor = lambda: data_v2
env = Environment(data_provider, summary_path='/tmp/ntucker_demo')
ntucker = NTUCKER_BCU(env)
args = NTUCKER_BCU.NTUCKER_Args(ranks=[10, 10, 10], validation_internal=10)
ntucker.build_model(args)
print("ntucker", type(ntucker))
ntucker.train(2000)
factor = ntucker.factors

##################################################################
# Test decomposition with clustering 
listAlgorithmes =["Kmeans++", "Skmeans","SpectralClustering","GMM"]
for a,algo in enumerate(listAlgorithmes):
    model_decompClustering = decomposition.DecompositionWithClustering(n_clusters = [3,3,2], modes = [1,2,3], algorithm = algo)
    model_decompClustering.fit(res_tucker)

    all_clustering = model_decompClustering.labels_

    mode1_clustering = all_clustering[0] 
    acc = np.around(accuracy(labels, mode1_clustering),3)
    print("Accuracy decompostion with "+ algo + " : ", acc)


##################################################################
#                 Visualization of results                       #
##################################################################

# Log-likelihood evolution 

plot_logLikelihood_evolution(model)

# Visualization of parameters evolution for poisson model
plot_gammaKK_evolution(model)

# Visualization of reorgnization of rows and columns for each slice
# based on obtained tensor co-clustering results 
plot_slice_reorganisation(data_v2,model)


