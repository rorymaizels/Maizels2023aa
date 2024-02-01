# general packages
import numpy as np
import pandas as pd
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

def velvet_pipeline(adata0, name):
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
        neighborhood_space="latent_space",
        biophysical_model = "full",
        gamma_mode = "learned",
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

def velvet_pipeline_with_smoothing(adata0, name):
    adata = adata0.copy()
    
    adata.layers['total'] = (
        adata.layers['total'].A if issparse(adata.layers['total']) else adata.layers['total']
    )
    adata.layers['new'] = (
        adata.layers['new'].A if issparse(adata.layers['new']) else adata.layers['new']
    )
    
    smoothing_cnx = vt.pp.connectivities(adata, n_neighbors=30)

    adata.layers['total_smooth'] = vt.pp.moments(X=adata.layers['total'],
        cnx=smoothing_cnx
    )

    adata.layers['new_smooth'] = vt.pp.moments(
        X=adata.layers['new'],
        cnx=smoothing_cnx
    )

    vt.pp.neighborhood(adata, n_neighbors=100)

    vt.ut.set_seed(0)
    
    vt.md.Velvet.setup_anndata(adata, x_layer='total_smooth', n_layer='new_smooth', knn_layer='knn_index')

    model = vt.md.Velvet(
        adata,
        n_latent = 50,
        linear_decoder = True,
        neighborhood_space="latent_space",
        biophysical_model = "full",
        gamma_mode = "learned",
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
    PIPELINE_NAME = 'velvet'
    pipeline = velvet_pipeline
    pipeline_smooth = velvet_pipeline_with_smoothing
    data = 'labelling'
    
    if data == 'splicing':
        bench_func = perform_benchmark_splicing
        other_func = perform_benchmark_other_splicing
        genes_func = gene_specific_benchmark_splicing
    elif data == 'labelling':
        bench_func = perform_benchmark
        other_func = perform_benchmark_other_labelling
        genes_func = gene_specific_benchmark
    
    bench_func(
        pipeline_name=f'{PIPELINE_NAME}_RAW',
        velocity_pipeline=pipeline, 
        output_folder='../../output_data/benchmarking_scores'
    )
    
    bench_func(
        pipeline_name=f'{PIPELINE_NAME}_SMOOTH',
        velocity_pipeline=pipeline_smooth, 
        output_folder='../../output_data/benchmarking_scores'
    )
    
    other_func(
        pipeline_name=f'{PIPELINE_NAME}_OMD',
        velocity_pipeline=pipeline, 
        output_folder='../../output_data/benchmarking_scores',
    )
    
    genes_func(
        output_folder='../../output_data/benchmarking_scores', 
        pipeline=pipeline, 
        pipeline_name=f'{PIPELINE_NAME}_GS'
    )
    
    print("")
    print("# # # # # # # # # # # # # # # # # # # # # # # # # ")
    print("~ ~ ~ ~ ~ ~ ~ BENCHMARKING COMPLETE ~ ~ ~ ~ ~ ~ ~ ")
    print(" # # # # # # # # # # # # # # # # # # # # # # # # #")