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
import unitvelo as utv

def unitvelo_pipeline(adata0, name):
    adata = adata0.copy()
    sc.pp.neighbors(adata)
    sc.tl.louvain(adata)
    os.system("mkdir -p TEMP_UNITVELO_OBJECT")
    adata.write_h5ad("TEMP_UNITVELO_OBJECT.h5ad")
    label='louvain'
    velo_config = utv.config.Configuration()
    velo_config.R2_ADJUST = False
    velo_config.VGENES = list(adata.var_names)
    velo_config.IROOT = None
    velo_config.FIT_OPTION = '1'
    velo_config.AGENES_R2 = 1
    velo_config.MIN_SHARED_COUNTS = 0
    velo_config.N_TOP_GENES = len(adata0.var_names)
    velo_config.N_PCs = 50
    velo_config.RESCALE_DATA = True
    velo_config.ASSIGN_POS_U = True

    velo_config.USE_RAW = False
    
    adata = utv.run_model("TEMP_UNITVELO_OBJECT.h5ad", label, config_file=velo_config)
    
    V = adata.layers['velocity']
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    
    os.system("rm TEMP_UNITVELO_OBJECT.h5ad")
    os.system("rm -r TEMP_UNITVELO_OBJECT")
    return V

def unitvelo_pipeline_without_smoothing(adata0, name):
    adata = adata0.copy()
    sc.pp.neighbors(adata)
    sc.tl.louvain(adata)
    os.system("mkdir -p TEMP_UNITVELO_OBJECT")
    adata.write_h5ad("TEMP_UNITVELO_OBJECT.h5ad")
    label='louvain'
    velo_config = utv.config.Configuration()
    velo_config.R2_ADJUST = False
    velo_config.VGENES = list(adata.var_names)
    velo_config.IROOT = None
    velo_config.FIT_OPTION = '1'
    velo_config.AGENES_R2 = 1
    velo_config.MIN_SHARED_COUNTS = 0
    velo_config.N_TOP_GENES = len(adata0.var_names)
    velo_config.N_PCs = 50
    velo_config.RESCALE_DATA = True
    velo_config.ASSIGN_POS_U = True

    velo_config.USE_RAW = True # only difference  
    
    adata = utv.run_model("TEMP_UNITVELO_OBJECT.h5ad", label, config_file=velo_config)
    
    V = adata.layers['velocity']
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    
    os.system("rm TEMP_UNITVELO_OBJECT.h5ad")
    os.system("rm -r TEMP_UNITVELO_OBJECT")
    return V

if __name__ == "__main__":
    PIPELINE_NAME = 'unitvelo'
    pipeline = unitvelo_pipeline_without_smoothing
    pipeline_smooth = unitvelo_pipeline
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