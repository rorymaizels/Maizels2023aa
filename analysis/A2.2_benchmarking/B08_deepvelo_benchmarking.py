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

import os

from benchmarking_functions import *

from deepvelo.utils import velocity, update_dict
from deepvelo.utils.preprocess import autoset_coeff_s
from deepvelo.utils.plot import statplot, compare_plot
from deepvelo import train, Constants
import deepvelo as dv

def deepvelo_pipeline(adata0, name):
    adata = adata0.copy()
    scv.pp.filter_genes(adata, min_shared_counts=1)
    scv.pp.moments(adata, n_neighbors=30, n_pcs=30)
    configs = {
        "name": "DeepVelo", # name of the experiment
        "loss": {"args": {"coeff_s": autoset_coeff_s(adata)}},
    }
    configs = update_dict(Constants.default_configs, configs)
    configs['n_gpu'] = 0
    
    trainer = dv.train(adata, configs)
    
    V = adata.layers['velocity']
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    
    V2 = np.zeros(adata0.shape)
    vn1, vn2 = adata0.var_names, adata.var_names

    for i, gene in enumerate(vn2):
        j = np.where(vn1==gene)[0][0]
        V2[:,j] = V[:,i]
    
    os.system('rm -r saved')
    return V2


if __name__ == "__main__":
    PIPELINE_NAME = 'deepvelo'
    pipeline = deepvelo_pipeline
    pipeline_smooth = deepvelo_pipeline
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