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

import velovi
from velovi import preprocess_data, VELOVI

def velovi_pipeline(adata0, name):
    print("RUNNING VELOVI WITH: ", name)
    adata = adata0.copy()
    scv.pp.log1p(adata)
    scv.pp.moments(adata, n_pcs=50, n_neighbors=30)
    
    adata.layers['Ms'] = np.array(adata.layers['Ms'].A if issparse(adata.layers['Ms']) 
                               else adata.layers['Ms'])

    adata.layers['Mu'] = np.array(adata.layers['Mu'].A if issparse(adata.layers['Mu']) 
                               else adata.layers['Mu'])
    
    adata = preprocess_data(
        adata,
        spliced_layer = "Ms",
        unspliced_layer = "Mu",
        filter_on_r2 = False
    )
    try:
        VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
        vae = VELOVI(adata)
        vae.train()
        V = vae.get_velocity(n_samples=25, velo_statistic="mean")
    except: # velovi encounters an issue of instability that leads to nans.
        V = np.zeros_like(adata.layers['Ms'])
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    return V

def velovi_pipeline_without_smoothing(adata0, name):
    print("RUNNING VELOVI WITH: ", name)
    adata = adata0.copy()
    scv.pp.log1p(adata)
    scv.pp.neighbors(adata)
    
    adata.layers['spliced'] = np.array(adata.layers['spliced'].A if issparse(adata.layers['spliced']) 
                               else adata.layers['spliced'])

    adata.layers['unspliced'] = np.array(adata.layers['unspliced'].A if issparse(adata.layers['unspliced']) 
                               else adata.layers['unspliced'])
    
    adata = preprocess_data(
        adata,
        spliced_layer = "spliced",
        unspliced_layer = "unspliced",
        filter_on_r2 = False
    )
    VELOVI.setup_anndata(adata, spliced_layer="spliced", unspliced_layer="unspliced")
    vae = VELOVI(adata)
    try:
        vae.train()
        V = vae.get_velocity(n_samples=25, velo_statistic="mean")
    except: # velovi encounters an issue of instability that leads to nans.
        V = np.zeros_like(adata.layers['spliced'])
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    return V

if __name__ == "__main__":
    PIPELINE_NAME = 'velovi'
    pipeline = velovi_pipeline_without_smoothing
    pipeline_smooth = velovi_pipeline
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