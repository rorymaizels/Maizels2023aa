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

import dynamo as dyn

def dynamo_pipeline(adata0, name):
    adata = adata0.copy()
    del adata.layers['old']
    adata.obs['labelling_time'] = 2.0

    dyn.pp.recipe_monocle(
        adata,
        tkey='labelling_time',
        genes_to_use=adata.var_names,
        keep_filtered_genes=True,
        normalized=True,
        num_dim=50,
    )
    clear_output(wait=False)

    dyn.tl.dynamics(
        adata
    )
    clear_output(wait=False)
    
    V = adata.layers['velocity_T']
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    
    return V

def dynamo_pipeline_without_smoothing(adata0, name):
    adata = adata0.copy()
    del adata.layers['old']
    adata.obs['labelling_time'] = 2.0

    dyn.pp.recipe_monocle(
        adata,
        tkey='labelling_time',
        genes_to_use=adata.var_names,
        keep_filtered_genes=True,
        normalized=True,
        num_dim=50,
    )
    clear_output(wait=False)
    
    # dynamo will break if you don't perform smoothing
    # but this is a work around to ensure smooth data 
    # is not used for main velocity calculation.
    dyn.tl.moments(adata)
    del adata.layers['M_t']
    del adata.layers['M_n']
    del adata.layers['M_tt']
    del adata.layers['M_tn']
    del adata.layers['M_nn']

    dyn.tl.dynamics(
        adata,
        model='deterministic',
        use_smoothed=False
    )
    clear_output(wait=False)
    
    V = adata.layers['velocity_T']
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    
    return V

if __name__ == "__main__":
    PIPELINE_NAME = 'dynamo'
    pipeline = dynamo_pipeline_without_smoothing
    pipeline_smooth = dynamo_pipeline
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