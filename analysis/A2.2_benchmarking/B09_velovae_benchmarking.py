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

path = '../../../../git/VeloVAE/'
import sys
sys.path.append(path)
import velovae as vv
import torch
import os

from benchmarking_functions import *

def velovae_pipeline(adata0, name):
    adata = adata0.copy()
    scv.pp.moments(adata, n_neighbors=30, n_pcs=30)

    torch.manual_seed(0)
    np.random.seed(0)
    vae = vv.VAE(adata, 
                 tmax=20, 
                 dim_z=5, 
                 device='cuda:0')
    
    config = {'test_iter':5}
    vae.train(adata,
              config=config,
              plot=False,
              embed=None)
    
    vae.save_anndata(adata, 'velovae', '.', file_name=f"deleteme.h5ad")
    os.system('rm deleteme.h5ad')
    
    V = adata.layers['velovae_velocity']
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    os.system('rm -r figures')
    return V
    

if __name__ == "__main__":
    PIPELINE_NAME = 'velovae'
    pipeline = velovae_pipeline
    pipeline_smooth = velovae_pipeline
    data = 'splicing'
    
    if data == 'splicing':
        bench_func = perform_benchmark_splicing
        other_func = perform_benchmark_other_splicing
        genes_func = gene_specific_benchmark_splicing
    elif data == 'labelling':
        bench_func = perform_benchmark
        other_func = perform_benchmark_other_labelling
        genes_func = gene_specific_benchmark
    
#     bench_func(
#         pipeline_name=f'{PIPELINE_NAME}_RAW',
#         velocity_pipeline=pipeline, 
#         output_folder='../../output_data/benchmarking_scores'
#     )
    
    bench_func(
        pipeline_name=f'{PIPELINE_NAME}_SMOOTH',
        velocity_pipeline=pipeline_smooth, 
        output_folder='../../output_data/benchmarking_scores'
    )
    
#     other_func(
#         pipeline_name=f'{PIPELINE_NAME}_OMD',
#         velocity_pipeline=pipeline, 
#         output_folder='../../output_data/benchmarking_scores',
#     )
    
    genes_func(
        output_folder='../../output_data/benchmarking_scores', 
        pipeline=pipeline, 
        pipeline_name=f'{PIPELINE_NAME}_GS'
    )
    
    print("")
    print("# # # # # # # # # # # # # # # # # # # # # # # # # ")
    print("~ ~ ~ ~ ~ ~ ~ BENCHMARKING COMPLETE ~ ~ ~ ~ ~ ~ ~ ")
    print(" # # # # # # # # # # # # # # # # # # # # # # # # #")