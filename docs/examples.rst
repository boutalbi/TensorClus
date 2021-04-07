Examples
========

The datasets used here are available at:

https://github.com/boutalbi/TensorClus/tree/master/TensorClus/reader

Basic usage
~~~~~~~~~~~

In the following example, the DBLP1 dataset is loaded from the reader module. A tensor co-clustering is applied using
the 'SparseTensorCoclusteringPoisson' algorithm with 3 clusters. The
The accuracy measure is printed and the predicted row labels and column labels are
retrieved for further exploration or evaluation.

.. code-block:: python

    import TensorClus.coclustering.sparseTensorCoclustering as tcSCoP
    from TensorClus.reader import load
    import numpy as np
    from coclust.evaluation.external import accuracy

    ##################################################################
    # Load DBLP1 dataset #
    ##################################################################
    data_v2, labels, slices = load.load_dataset("DBLP1_dataset")
    n = data_v2.shape[0]
    ##################################################################
    # Execute TSPLBM on the dataset #
    ##################################################################

    # Define the number of clusters K
    K = 3
    # Optional: initialization of rows and columns partitions
    z=np.zeros((n,K))
    z_a=np.random.randint(K,size=n)
    z=np.zeros((n,K))+ 1.e-9
    z[np.arange(n) , z_a]=1
    w=np.asarray(z)

    # Run TSPLBM

    model = tcSCoP.SparseTensorCoclusteringPoisson(n_clusters=K , fuzzy = True,init_row=z, init_col=w,max_iter=50)
    model.fit(data_v2)
    predicted_row_labels = model.row_labels_
    predicted_column_labels = model.column_labels_

    acc = np.around(accuracy(labels, predicted_row_labels),3)
    print("Accuracy : ", acc)