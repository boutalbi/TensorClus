# -*- coding: utf-8 -*-

"""

The :mod:`TensorClus.vizualisation` module provides functions to visualize
different measures or data.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import matplotlib

logger = logging.getLogger(__name__)
plt.style.use('ggplot')


def generateColour():
    """
    Generate random color.

    Parameters
    ----------

    Returns
    -------
    str
        hex color
    """
    _HEX = list('0123456789ABCDEF')
    return '#' + ''.join(_HEX[randint(0, len(_HEX)-1)] for _ in range(6))

def duplicates(lst, item):
    """ 
    Find index of duplicated values.
    
    Parameters
    ----------
    lst: list of values
    item: values to determine
    
    Returns
    -------
    list
        index of dipulicated values
    """
    return [i for i, x in enumerate(lst) if x == item]

def Plot_CoClust_axes_etiquette(title, fig, axes, data, phiR, phiC, K, L, etiquette):
    """ 
    Plot CoClustering results for each slice on specific axes.

    Parameters
    ----------
    title: title of figure

    fig  : figure that includes all axes

    axes : list of axes corresponding to the number of slices

    data : tensor data

    phiR : row clustering partition

    phiC : row clustering partition

    K    : number of row cluster

    L    : number of columns cluster

    etiquette : name of slices

    """
    n = data.shape[0]
    d = data.shape[1]
    ### Order rows
    infR = int(phiR.min())
    indexR = list(np.asarray(duplicates(phiR, infR)))
    indexAffichagesLignes = [np.count_nonzero(phiR == infR)]
    for r in range(infR + 1, K + infR):
        indexAffichagesLignes.append(np.count_nonzero(phiR == r) + indexAffichagesLignes[r - 2])
        indexR.extend(duplicates(phiR, r))

    Newdata = data[indexR, :]
    ### Order columns
    infC = int(phiC.min())
    indexC = list(np.asarray(duplicates(phiC, infC)))
    indexAffichagesColonnes = [np.count_nonzero(phiC == infC)]
    for c in range(infC + 1, L + infC):
        indexAffichagesColonnes.append(np.count_nonzero(phiC == c) + indexAffichagesColonnes[c - 2])
        indexC.extend(duplicates(phiC, c))

    Newdata1 = Newdata[:, indexC]
    ##################################################################################
    # levels contient le numero de la ligne qu'on souhaite afficher
    levels = np.unique(np.asarray(indexAffichagesLignes))
    levels = np.asarray(list(levels[i] for i in range(0, len(levels) - 1)))
    # levels1 contient le numero de la colonne qu'on souhaite afficher
    levels1 = np.unique(np.asarray(indexAffichagesColonnes))
    levels1 = np.asarray(list(levels1[i] for i in range(0, len(levels1) - 1)))
    ##################################################################################

    #im2 = axes.pcolormesh(Newdata1, cmap='Oranges', rasterized=True)
    im2 = axes.spy(Newdata1, markersize=0.5, color="black", aspect='auto')

    axes.set_title(title, fontsize=16)
    axes.set_xticks(levels1)
    axes.set_xticklabels(np.asarray(levels1).astype(str))
    axes.set_yticks(levels)
    axes.set_yticklabels(np.asarray(levels).astype(str))

    ###################################################################################
    cl = np.array([list(range(0, d + 1))])
    xcontour = np.concatenate((cl, cl), axis=0).T
    ycontour = np.array([[0, n], ] * len(xcontour))
    zcontour = np.array([[0, n], ] * len(xcontour))

    cc = np.array([list(range(0, n + 1))])
    ycontour1 = np.concatenate((cc, cc), axis=0).T
    xcontour1 = np.array([[0, d], ] * len(ycontour1))
    zcontour1 = np.array([[0, d], ] * len(ycontour1))
    ###################################################################################

    axes.contour(xcontour, ycontour, zcontour, linewidths=3, colors="black",
                 levels=levels)
    axes.contour(xcontour1, ycontour1, zcontour1, linewidths=3, colors="black",
                 levels=levels1)

def plot_logLikelihood_evolution(model, do_plot=True, save=False, dpi = 200):
    """
    Plot all intermediate loglikelihood for a model at each iteration.

    Parameters
    ----------
    model: :class:`TensorClus.coclustering`, Fitted model

    do_plot: boolean, Whether the plot should be displayed. True by default. Disabling this allows users to handle displaying the plot themselves.

    save   : boolean, False by default. Allowing save plot as image

    dpi    : int, 200 by  default. Allowing to choose a specific resolution when saving image

    """

    # Prepare a subplot and set the axis tick values and labels
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1, (len(model.criterions) + 1), 1)
    plt.xticks(np.arange(0, len(model.criterions) + 1, 1))
    ax.set_xticklabels(labels)

    # Plot all intermdiate modularities
    plt.plot(model.criterions, marker='o')

    # Set the axis titles
    plt.ylabel("Log-Likelihood", size=10)
    plt.xlabel("Iterations", size=10)

    # Set the main plot titlee
    plt.title("\nEvolution of Log-Likelihood\n", size=12)

    # Remove automatic ticks
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')

    # Plot a dashed horizontal line around max modularity
    plt.axhline(max(model.criterions), linestyle="dashed")
    plt.axhline((max(model.criterions) - model.tol), linestyle="dashed")
    if do_plot:
        plt.show()

    if save :
        plt.savefig('evolution_loglikelihood.png', format='png', dpi=dpi)



def plot_parameter_evolution(model, do_plot=True, save=False, dpi = 200):
    """
    Plot all intermediate gammaKK parameters for a model at each iteration.

    Parameters
    ----------
    model: :class:`TensorClus.coclustering`, Fitted model

    do_plot: boolean, Whether the plot should be displayed. True by default. Disabling this allows users to handle displaying the plot themselves.

    save   : boolean, False by default. Allowing save plot as image

    dpi    : int, 200 by  default. Allowing to choose a specific resolution when saving image
    """

    # Prepare a subplot and set the axis tick values and labels
    if str(type(model).__name__) == "SparseTensorCoclusteringPoisson":
        K = model.n_clusters
        L = int(K)

        colors =  [generateColour() for i in range(K*L)]
        tab_evol_gammakl_VEM = model.gamma_kl_evolution
    elif str(type(model).__name__) == "TensorCoclusteringPoisson":
        K = model.n_row_clusters
        L = model.n_col_clusters
        colors =  [generateColour() for i in range(K*L)]
        tab_evol_gammakl_VEM = model.gamma_kl_evolution
    else :
        K = model.n_row_clusters
        L = model.n_col_clusters

        colors = [generateColour() for i in range(K * L)]
        tab_evol_gammakl_VEM = model.mu_kl_evolution

    fig, axes = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(1, (len(model.criterions) + 1), 1)
    plt.xticks(np.arange(0, len(model.criterions) + 1, 1))
    axes.set_xticklabels(labels)

    cpt = 0
    for k in range(K):
        for l in range(L):
            plt.plot(tab_evol_gammakl_VEM[k][l], c=colors[cpt], marker='.', linewidth=1.5,label ='block ('+str(k + 1)+',' + str(l + 1)+')')

            cpt = cpt + 1

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.8, wspace=0.1)
    axes.legend(bbox_to_anchor=(1.21, 1.02))



    # Set the axis titles
    plt.xlabel("Iterations", size=12, fontweight='bold')
    plt.ylabel('Parameters per block (k,l)', size=12)    

    # Set the main plot titlee
    plt.title('Evolution of parameters', size=12)

    # Remove automatic ticks
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right='off', left='off')

    # Plot a dashed horizontal line around max modularity
    plt.axhline(np.amax(tab_evol_gammakl_VEM), linestyle="dashed")

    if do_plot:
        plt.show()

    if save :
        plt.savefig('evolution_gammaKK.png', format='png', dpi=dpi)


def plot_slice_reorganisation(data,model, slicesName = None,  do_plot=True, save=False, dpi = 200):
    """
    Plot all intermediate modularities for a model.

    Parameters
    ----------
    data   : tensor data

    model: :class:`TensorClus.coclustering.CoclustMod`, Fitted model

    slicesName : list of slice names

    do_plot: boolean, Whether the plot should be displayed. True by default. Disabling this allows users to handle displaying the plot themselves.

    save   : boolean, False by default. Allowing save plot as image

    dpi    : int, 200 by  default. Allowing to choose a specific resolution when saving image
    """

    # Prepare a subplot and set the axis tick values and labels


    K = model.n_clusters
    L = int(K)
    n = data.shape[0]
    d = data.shape[1]
    v = data.shape[2]
    phiR = model.row_labels_
    phiC = model.column_labels_
    ################################################
    # Compute number of row and column subplots
    sqrtV = np.sqrt(v)
    if (v % sqrtV) == 0 :
       r = int(sqrtV)
       c = int(v/sqrtV)
       casesVides = 0
    else :
        r = int(sqrtV)
        if (r*r)< v :
            c = int(r)
            casesVides = (r*r) - v
        else :
            c = r+1
            casesVides = (r * (r+1)) - v

    ################################################
    fig, axes = plt.subplots(r, c, sharey=True)

    cpt_plot_row = 0
    cpt_plot_col = 0
    if slicesName is None:
        features_liste = ["slice "+str((b+1)) for b in range(v)]
        features_liste = np.asarray(features_liste)
    else:
        features_liste = np.asarray(slicesName)

    for fe in range(v):

        y_true = data[:, :, fe].reshape(n, d)

        if cpt_plot_col == r:
            cpt_plot_col = 0
            cpt_plot_row = cpt_plot_row + 1

        Plot_CoClust_axes_etiquette(features_liste[fe], fig, axes[cpt_plot_row, cpt_plot_col], y_true,
                                                     phiR, phiC, K, L, features_liste.tolist())

        cpt_plot_col = cpt_plot_col + 1

    for cv in range(casesVides) :
        cs = cv+1
        axes[-1, (-1*cs)].axis('off')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 9)
    plt.subplots_adjust(top=0.95, bottom=0.04, left=0.08, right=0.96, hspace=0.45)


    if do_plot:
        plt.show()

    if save :
        plt.savefig('evolution_gammaKK.png', format='png', dpi=dpi)
