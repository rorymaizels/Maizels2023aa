import velvet as vt

# general packages
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse

# velocity packages
import scanpy as sc
import scvelo as scv
import anndata as ann

# plotting packages
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
from IPython.display import clear_output

# color palette object
from colors import colorpalette as colpal

from benchmarking_functions import *

import velvet as vt
import torch

def velvet_pipeline(adata0, name, NC='latent_space', DG='learned'):
    adata = adata0.copy()

    adata.layers['total'] = (
        adata.layers['total'].A if issparse(adata.layers['total']) else adata.layers['total']
    )
    adata.layers['new'] = (
        adata.layers['new'].A if issparse(adata.layers['new']) else adata.layers['new']
    )
    
    vt.pp.neighborhood(adata, n_neighbors=100)
    
    vt.ut.set_seed(0)
    
    vt.md.Velvet.setup_anndata(adata, x_layer='total', n_layer='new', knn_layer='knn_index')

    model = vt.md.Velvet(
        adata,
        n_latent = 50,
        linear_decoder = True,
        neighborhood_space = NC,
        biophysical_model = "full",
        gamma_mode = DG,
        labelling_time = 2.0,
    )

    model.setup_model()
    
    model.train(
        batch_size = adata.shape[0],
        max_epochs = 1000, 
        freeze_vae_after_epochs = 200,
        constrain_vf_after_epochs = 200,
        lr=0.001,
    )
    
    V = model.predict_velocity()
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    
    return V

if __name__ == "__main__":
    for name, nc, dg in zip(
        ["velvet_NoC_NoG","velvet_NoC_WiG","velvet_WiC_NoG","velvet_HDC_WiG"],
        ["none","none","latent_space","gene_space"],
        ["fixed","learned","fixed","learned"],
    ):
        print(f"Running {name}...")
        def func(adata, name):
            return velvet_pipeline(adata, name, NC=nc, DG=dg)

        perform_benchmark(
            pipeline_name=name,
            velocity_pipeline=func, 
            output_folder='../../output_data/benchmarking_scores'
        )

    print("")
    print("# # # # # # # # # # # # # # # # # # # # # # # # # ")
    print("~ ~ ~ ~ ~ ~ ~ BENCHMARKING COMPLETE ~ ~ ~ ~ ~ ~ ~ ")
    print(" # # # # # # # # # # # # # # # # # # # # # # # # #")