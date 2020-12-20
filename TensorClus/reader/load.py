# -*- coding: utf-8 -*-

"""

The :mod:`TensorClus.reader` module provides functions to load and read
different data fprmat.
"""

import pandas as pd
from random import randint
import logging
import os
import numpy as np
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext

logger = logging.getLogger(__name__)


def load_dataset(datasetName):
    """ Load one of availble dataset.
    
    Parameters
    ----------
    datasetName : str
        the name of dataset
    Returns
    -------
    tensor
        three-way numpy array
    labels
        true row classes
    slices
        slices name
    """
    base_dir = dirname(__file__)
    print("base_dir",base_dir)
    datasetName = datasetName + ".npz"
    outfile = join(base_dir , datasetName)
    npzfile = np.load(outfile)
    tensor  = npzfile['arr_0']
    labels  = npzfile['arr_1']
    slices  = npzfile['arr_2']
    return tensor, labels, slices

def read_txt_tensor(filePath):
    """ read tensor data from text file.
    
    Parameters
    ----------
    filePath : str
        the path of file
    Returns
    -------
    tensor
        three-way numpy array
    """
    tensor  = pd.read_csv(filePath,  header=0, names= ["v1", "v2", "v3", "v4"])
    rowIndices = np.asarray(tensor.v1).reshape(tensor.shape[0]).astype(int)
    colIndices = np.asarray(tensor.v2).reshape(tensor.shape[0]).astype(int)
    vIndices   = np.asarray(tensor.v3).reshape(tensor.shape[0]).astype(int)
    values     = np.asarray(tensor.v4).reshape(tensor.shape[0])

    maxRow  = int(np.amax(rowIndices))+1
    maxCol  = int(np.amax(colIndices))+1
    maxV    = int(np.amax(vIndices))  +1

    tensorData = np.zeros((maxRow, maxCol, maxV))
    tensorData[rowIndices, colIndices, vIndices] = values

    return tensorData

def save_txt_tensor(tensor, fileName):
    """ save tensor data as a text file.
    
    Parameters
    ----------
    tensor   : tensor array
    filePath : str
        the path of file
    """
    nrow, ncol, v = tensor.shape
    rowIndices, colIndices, vIndices   = np.indices((nrow, ncol, v))
    rowIndices = rowIndices.reshape(nrow *  ncol * v)
    colIndices = colIndices.reshape(nrow * ncol * v)
    vIndices = vIndices.reshape(nrow * ncol * v)
    values          =tensor[rowIndices, colIndices, vIndices ].reshape(nrow * ncol * v)
    data            = np.zeros((len(values),  4))
    data[:, 0]      = rowIndices
    data[:, 1]      = colIndices
    data[:, 2]      = vIndices
    data[:, 3]      = values

    np.savetxt(fileName, data, delimiter=',')
