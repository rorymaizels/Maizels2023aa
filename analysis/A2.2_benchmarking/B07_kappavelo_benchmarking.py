import velvet as vt

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

import sys
sys.path.insert(0, '../../../../git/velocity_package/')
import velocity

def kvelo_pipeline(adata0, name):
    adata = adata0.copy()
    scv.pp.moments(adata, n_pcs=50, n_neighbors=30)
    velocity.tl.fit.recover_reaction_rate_pars(adata, use_raw=False)
    velocity.tl.fit.get_velocity(adata, use_raw=False, normalise=None)
    
    V = adata.layers['velocity']
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    return V

def kvelo_pipeline_without_smoothing(adata0, name):
    adata = adata0.copy()
    fix = lambda x: x.A if issparse(x) else x
    adata.layers['unspliced'] = fix(adata.layers['unspliced'])
    adata.layers['spliced'] = fix(adata.layers['spliced'])
    velocity.tl.fit.recover_reaction_rate_pars(adata, use_raw=True)
    velocity.tl.fit.get_velocity(adata, use_raw=True, normalise=None)
    
    V = adata.layers['velocity']
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    return V

if __name__ == "__main__":
    PIPELINE_NAME = 'kappavelo'
    pipeline = kvelo_pipeline_without_smoothing
    pipeline_smooth = kvelo_pipeline
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