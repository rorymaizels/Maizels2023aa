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
import os
from pyrovelocity.api import train_model
from pyrovelocity.plot import vector_field_uncertainty

def pyrovelocity_pipeline(adata0, name):
    adata = adata0.copy()
    try:
        adata.layers['raw_spliced']   = adata.uns['raw_spliced_counts'] #.astype(int)
        adata.layers['raw_unspliced'] = adata.uns['raw_unspliced_counts'] #.astype(int)
        adata.obs['u_lib_size_raw'] = adata.layers['raw_unspliced'].toarray().sum(-1)
        adata.obs['s_lib_size_raw'] = adata.layers['raw_spliced'].toarray().sum(-1)
    except:
        print("###")
        print("FAILURE!!!")
        print(name)
        print(adata)
        print("###")
        raise TypeError("FAILURE!!!!!")
    num_epochs = 1000 # large data
    # Model 1
    adata_model_pos = train_model(adata,
                                   max_epochs=num_epochs, svi_train=True, log_every=100,
                                   patient_init=45,
                                   batch_size=4000, use_gpu=0, cell_state='state_info',
                                   include_prior=False,
                                   offset=False,
                                   library_size=True,
                                   patient_improve=1e-3,
                                   model_type='auto',
                                   guide_type='auto_t0_constraint',
                                   train_size=1.0)

    trained_model = adata_model_pos[0]
    posterior_samples = adata_model_pos[1]

    V = (
        posterior_samples['ut'] * posterior_samples["beta"]
        - posterior_samples["st"] * posterior_samples["gamma"]
    ).mean(0)
    
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    return V

if __name__ == "__main__":
    PIPELINE_NAME = 'pyrovelocity'
    pipeline = pyrovelocity_pipeline
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
    
    # bench_func(
    #     pipeline_name=f'{PIPELINE_NAME}_SMOOTH',
    #     velocity_pipeline=pipeline_smooth, 
    #     output_folder='../../output_data/benchmarking_scores'
    # )
    
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