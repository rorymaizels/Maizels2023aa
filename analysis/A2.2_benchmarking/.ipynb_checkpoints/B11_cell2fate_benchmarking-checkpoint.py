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

import cell2fate as c2f

def cell2fate_pipeline(adata0, name):
    adata = adata0.copy()

    adata.layers['spliced'] = (
        adata.layers['spliced'].A if issparse(adata.layers['spliced']) else adata.layers['spliced']
    )
    adata.layers['unspliced'] = (
        adata.layers['unspliced'].A if issparse(adata.layers['unspliced']) else adata.layers['unspliced']
    )
    
    c2f.Cell2fate_DynamicalModel_PreprocessedCounts.setup_anndata(
        adata, 
        spliced_label='spliced', 
        unspliced_label='unspliced'
    )
    n_modules = c2f.utils.get_max_modules(adata)

    mod = c2f.Cell2fate_DynamicalModel_PreprocessedCounts(adata, n_modules = n_modules)
    mod.train()
    mod.export_posterior(adata)
    mod.compute_and_plot_total_velocity_scvelo(adata, delete = False, plot = False)
    
    V = adata.layers['velocity']
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    return V

def cell2fate_pipeline_with_smoothing(adata0, name):
    adata = adata0.copy()

    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    
    adata.layers['Ms'] = (
        adata.layers['Ms'].A if issparse(adata.layers['Ms']) else adata.layers['Ms']
    )
    adata.layers['Mu'] = (
        adata.layers['Mu'].A if issparse(adata.layers['Mu']) else adata.layers['Mu']
    )
    
    c2f.Cell2fate_DynamicalModel_PreprocessedCounts.setup_anndata(
        adata, 
        spliced_label='Ms', 
        unspliced_label='Mu'
    )
    n_modules = c2f.utils.get_max_modules(adata)

    mod = c2f.Cell2fate_DynamicalModel_PreprocessedCounts(adata, n_modules = n_modules)
    mod.train()
    mod.export_posterior(adata)
    mod.compute_and_plot_total_velocity_scvelo(adata, delete = False, plot = False)
    
    V = adata.layers['velocity']
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    return V

if __name__ == "__main__":
    PIPELINE_NAME = 'cell2fate'
    pipeline = cell2fate_pipeline
    pipeline_smooth = cell2fate_pipeline_with_smoothing
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