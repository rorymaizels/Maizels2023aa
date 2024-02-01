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

path = '../../../../git/LatentVelo/'
import sys
import os
sys.path.append(path)
import latentvelo as ltv

def latentvelo_pipeline(adata0, name):
    adata = adata0.copy()

    adata.layers['spliced'] = (adata.layers['spliced'].A if issparse(adata.layers['spliced'])
                               else adata.layers['spliced'])
    adata.layers['unspliced'] = (adata.layers['unspliced'].A if issparse(adata.layers['unspliced'])
                               else adata.layers['unspliced'])
    
    ltv.utils.standard_clean_recipe(
        adata, 
        spliced_key='spliced', 
        unspliced_key='unspliced',
        normalize_library=False,
        smooth = True, 
        log=False
    )
    spliced_library_sizes = adata.layers['spliced'].sum(1)
    unspliced_library_sizes = adata.layers['unspliced'].sum(1)

    adata.obs['spliced_size_factor'] = spliced_library_sizes #spliced_all_size_factors
    adata.obs['unspliced_size_factor'] = unspliced_library_sizes #unspliced_all_size_factors
    adata.var['velocity_genes'] = True

    try:
        model = ltv.models.VAE(observed = adata.shape[1])
    
        epochs, val_ae, val_traj = ltv.train(
            model,
            adata,
            name='delete'
        )
        
        latent_adata, adata = ltv.output_results(
            model, 
            adata,
            gene_velocity = True,
            decoded = True,
            embedding='pca')
        V = adata.layers['velo'] # we want high-dimensional velocity
    except AssertionError:
        V = np.zeros_like(np.asarray(adata.layers['spliced']))
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    os.system('rm -r delete')
    return V

def latentvelo_pipeline_without_smoothing(adata0, name):
    adata = adata0.copy()
    
    adata.layers['spliced'] = (adata.layers['spliced'].A if issparse(adata.layers['spliced'])
                               else adata.layers['spliced'])
    adata.layers['unspliced'] = (adata.layers['unspliced'].A if issparse(adata.layers['unspliced'])
                               else adata.layers['unspliced'])
    
    ltv.utils.standard_clean_recipe(
        adata, 
        spliced_key='spliced', 
        unspliced_key='unspliced',
        normalize_library=False,
        smooth = False, 
        log=False
    )
    spliced_library_sizes = adata.layers['spliced'].sum(1)
    unspliced_library_sizes = adata.layers['unspliced'].sum(1)

    adata.obs['spliced_size_factor'] = spliced_library_sizes #spliced_all_size_factors
    adata.obs['unspliced_size_factor'] = unspliced_library_sizes #unspliced_all_size_factors
    adata.var['velocity_genes'] = True
    try:
        model = ltv.models.VAE(observed = adata.shape[1])
    
        epochs, val_ae, val_traj = ltv.train(
            model,
            adata,
            name='delete'
        )
        
        latent_adata, adata = ltv.output_results(
            model, 
            adata,
            gene_velocity = True,
            decoded = True,
            embedding='pca')
        V = adata.layers['velo'] # we want high-dimensional velocity
    except AssertionError:
        V = np.zeros_like(np.asarray(adata.layers['spliced']))
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    os.system('rm -r delete')
    return V

if __name__ == "__main__":
    PIPELINE_NAME = 'latentvelo'
    pipeline = latentvelo_pipeline_without_smoothing
    pipeline_smooth = latentvelo_pipeline
    data = 'splicing'
    
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