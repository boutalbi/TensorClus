# TensorClus

TensorClus (Tensor Clustering) is a first Python library aiming to clustering and co-clustering of tensor data.
It allows to easily perform tensor clustering trought decomposition or tensor learning and tensor algebra. 
TensorClus allows easy interaction with other python packages such as NumPy, Tensorly, TensorFlow or TensorD, and run methods at scale on CPU or GPU.

**It supports major operating systems namely Microsoft Windows, macOS, and Ubuntu**.

[![N|Solid](https://github.com/boutalbi/TensorClus/blob/master/BinaryTensorData.PNG?raw=true)](https://link.springer.com/article/10.1007/s41060-020-00205-5)

- Source-code: https://github.com/boutalbi/TensorClus
- Jupyter Notebooks: https://github.com/boutalbi/TensorClus/blob/master/demo_tensorClus.ipynb

### Brief description 
TensorClus library provides multiple functionalities:
- Several datasets 
- Tensor co-clustering with various data type
- Tensor decomposition and clustering
- Visualization

### Requirements
```python
numpy==1.18.3
pandas==1.0.3
scipy==1.4.1
matplotlib==3.0.3
scikit-learn==0.22.2.post1
coclust==0.2.1
tensorD==0.1
tensorflow==2.3.0
tensorflow-gpu==2.3.0
tensorflow-estimator==2.3.0
tensorly==0.4.5
```

### Installing TensorClus
For installing TensorClus package use the following command
```
pip install -U TensorClus
```

To clone TensorClus project from github
```
# Install git LFS via https://www.atlassian.com/git/tutorials/git-lfs
# initialize Git LFS
git lfs install Git LFS initialized.
git init Initialized
# clone the repository
git clone https://github.com/boutalbi/TensorClus.git
cd TensorClus
# Install in editable mode with `-e` or, equivalently, `--editable`
pip install -e .
```
For more details about TensorClus, see [Documentation](https://tensorclus.readthedocs.io/en/latest/).

### License
TensorClus is released under the MIT License (refer to LISENSE file for details).

### Examples

```python
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
```
### Datasets

The following datasets and their [description](https://github.com/boutalbi/TensorClus/blob/master/data_description.md) are available in Google Drive. 

- [DBLP1 dataset](https://shorturl.at/ayBG8)
- [DBLP2 dataset](https://shorturl.at/fnt37)
- [PubMed Diabets-4K dataset](https://shorturl.at/rDUY2)
- [Nus-Wide-8 dataset](https://shorturl.at/abK17)


<!---
### Citing
If you use TensorClus in an academic paper, please cite
```
@article{boutalbi2020tensor,
 title={Tensor latent block model for co-clustering},
 author={Boutalbi, Rafika and Labiod, Lazhar and Nadif, Mohamed},
 journal={International Journal of Data Science and Analytics},
 pages={1--15},
 year={2020},
 publisher={Springer},
 doi= {10.1007/s41060-020-00205-5},
 url= "https://link.springer.com/article/10.1007/s41060-020-00205-5"
}
```
-->

### References
[1] Boutalbi, R., Labiod, L., & Nadif, M. (2020). Tensor latent block model for co-clustering. International Journal of Data Science and Analytics, 1-15.

[2] Boutalbi, R., Labiod, L., & Nadif, M. (2019, July). Sparse Tensor Co-clustering as a Tool for Document Categorization. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 1157-1160).

[3] Boutalbi, R., Labiod, L., & Nadif, M. (2019, April). Co-clustering from Tensor Data. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp. 370-383). Springer.
