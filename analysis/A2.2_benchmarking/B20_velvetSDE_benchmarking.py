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

import velvet as vt

def velvetSDE_pipeline(adata0, name):
    adata = adata0.copy()
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
    
    model.module = model.module.to('cuda')
    
    model.get_latent_dynamics(return_data=False)
    
    mp = vt.sb.MarkovProcess(
        model,
        n_neighbors=10,
        use_space='latent_space',
        use_spline=True,
        use_similarity=False,
    )

    sde = vt.sb.SDE(
        model.module.n_latent,
        prior_vectorfield=model.module.vf,
        noise_scalar=0.15,
        device=model.device
    )

    model.adata.obs['index'] = np.arange(model.adata.shape[0])
    vt.sm.VelvetSDE.setup_anndata(
        model, 
        x_layer='total', 
        index_key='index'
    )

    sde_model = vt.sm.VelvetSDE(
        model,
        sde,
        mp,
    )

    sde_model.train(
        max_epochs = 250,
        n_trajectories = 200,
        n_simulations = 50,
        n_steps = 30,
        n_markov_steps=15,
        t_max=25,
        dt = 1.0,
        lr = 0.001,
    )    
    
    X = model.adata_manager.get_from_registry("X")
    X = X.A if issparse(X) else X
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.tensor(X, device=torch_device)
    b = torch.zeros(X.shape[0], device=torch_device)
    model.module.to(torch_device)
    with torch.no_grad():
        inf = model.module.inference(x, b) # model and sde_model have same VAE
        z = inf['z']
        vz = sde_model.module.sde.drift(z)
        gen = model.module.generative(
                z,
                vz,
                inf['library'],
                model.module.t,
                b
        )
    V = gen['vel'].detach().cpu().numpy()
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    return V

def velvetSDE_pipeline_with_smoothing(adata0, name):
    adata = adata0.copy()
    
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
    
    model.module = model.module.to('cuda')

    model.get_latent_dynamics(return_data=False)
    
    mp = vt.sb.MarkovProcess(
        model,
        n_neighbors=10,
        use_space='latent_space',
        use_spline=True,
        use_similarity=False,
    )

    sde = vt.sb.SDE(
        model.module.n_latent,
        prior_vectorfield=model.module.vf,
        noise_scalar=0.15,
        device=model.device
    )

    model.adata.obs['index'] = np.arange(model.adata.shape[0])
    vt.sm.VelvetSDE.setup_anndata(
        model, 
        x_layer='total_smooth', 
        index_key='index'
    )

    sde_model = vt.sm.VelvetSDE(
        model,
        sde,
        mp,
    )

    sde_model.train(
        max_epochs = 250,
        n_trajectories = 200,
        n_simulations = 50,
        n_steps = 30,
        n_markov_steps=15,
        t_max=25,
        dt = 1.0,
        lr = 0.001,
    )    
    
    X = model.adata_manager.get_from_registry("X")
    X = X.A if issparse(X) else X
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.tensor(X, device=torch_device)
    b = torch.zeros(X.shape[0], device=torch_device)
    model.module.to(torch_device)
    with torch.no_grad():
        inf = model.module.inference(x, b) # model and sde_model have same VAE
        z = inf['z']
        vz = sde_model.module.sde.drift(z)
        gen = model.module.generative(
                z,
                vz,
                inf['library'],
                model.module.t,
                b
        )
    V = gen['vel'].detach().cpu().numpy()
    V = V.A if issparse(V) else V
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    return V

if __name__ == "__main__":
    PIPELINE_NAME = 'velvetSDE'
    pipeline = velvetSDE_pipeline
    pipeline_smooth = velvetSDE_pipeline_with_smoothing
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