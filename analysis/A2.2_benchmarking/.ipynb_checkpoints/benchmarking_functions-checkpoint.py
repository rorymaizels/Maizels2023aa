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

# we implement unitvelo's evaluation 
# originally from https://github.com/StatBiomed/UniTVelo/blob/main/unitvelo/eval_utils.py
# paper: https://www.nature.com/articles/s41467-022-34188-7
# authors: Mingze Gao, Chen Qiao & Yuanhua Huang 

from eval_functions import unitvelo_cross_boundary_correctness as cross_boundary_correctness
from  eval_functions import unitvelo_inner_cluster_coh as inner_cluster_coh

from sklearn.metrics.pairwise import cosine_similarity

# the object that will contain the data and data-specific parameters for benchmarking

class BenchMarkingData:
    def __init__(self, name, func, pt=True, scifate2=True):
        self.name = name
        adata = sc.read_h5ad(f'../../data/benchmarking/{name}.h5ad')
        
        self.adata = prepare_for_test(
            adata,
            name,
            func,
            pt=pt,
            scifate2=scifate2
        )
        
        self.cluster_edges()
        
    def cluster_edges(self):
        name = self.name.replace("splicing_","")
        if name == "mini_MN":
            self.obs = 'leiden'
            self.cluster_edges = [
                ('0','2'),
                ('2','4'),
                ('2','3'),
                ('4','5'),
            ]
        elif name == "mini_V3":
            self.obs = 'leiden'
            self.cluster_edges = [
                ('0','1'),
                ('1','3'),
                ('3','2'),
                ('2','4')
            ]
        elif name == "mini_MD":
            self.obs = 'leiden'
            self.cluster_edges = [
                ('2','3'),
                ('3','0'),
                ('3','5'),
                ('0','1'),
                ('1','4'),
            ]
        elif name == "midi_NM":
            self.obs = 'cell_annotation'
            self.cluster_edges = [
                ('Early_Neural','Neural'),
                ('NMP','Early_Neural'),
                ('NMP','Mesoderm')
            ]
        elif name == 'midi_Ne':
            self.obs = 'cell_annotation'
            self.cluster_edges = [
                ('Neural','pMN'),
                ('pMN','MN'),
                ('pMN','p3'),
                ('p3','V3')
            ]
        elif name == 'maxi':
            self.obs = 'cell_annotation'
            self.cluster_edges = [
                ('Early_Neural','Neural'),
                ('NMP','Early_Neural'),
                ('NMP','Mesoderm'),
                ('Neural','pMN'),
                ('pMN','MN'),
                ('pMN','p3'),
                ('p3','V3')
            ]
            
        elif name == "dentategyrus_lamanno":
            self.obs = 'clusters'
            self.cluster_edges = [
                ('Nbl1','Nbl2'),
                ('Nbl2','ImmGranule1'),
                ('Nbl2','CA'),
                ('CA','CA2-3-4'),
                ('ImmGranule1','ImmGranule2'),
                ('ImmGranule2','Granule')
            ]

        elif name == "dentategyrus":
            self.obs = 'clusters'
            self.cluster_edges = [
                ('Neuroblast','Granule immature'),
                ('Neuroblast','Granule mature')
            ]

        elif name == "forebrain":
            self.obs = 'clusters'
            self.cluster_edges = [
                ('0','1'),
                ('1','2'),
                ('2','3'),
                ('3','4'),
                ('4','5'),
                ('5','6')
            ]

        elif name == "gastrulation_erythroid":
            self.obs = 'celltype'
            self.cluster_edges = [
                ('Blood progenitors 1','Blood progenitors 2'),
                ('Blood progenitors 2','Erythroid1'),
                ('Erythroid1','Erythroid2'),
                ('Erythroid2','Erythroid3'),
            ]

        elif name == "mouse_motor_neuron":
            self.obs = 'leiden'
            self.cluster_edges = [
                ('3','2'),
                ('4','0'),
                ('0','1'),
                ('1','5')
            ]

        elif name == "pancreas":
            self.obs = 'clusters_coarse'
            self.cluster_edges = [
                ('Ductal','Ngn3 low EP'),
                ('Ngn3 low EP','Ngn3 high EP'),
                ('Ngn3 high EP','Pre-endocrine'),
                ('Pre-endocrine','Endocrine')
            ]

        elif name == "scifate_benchmark":
            self.obs = 'treatment_time'
            self.cluster_edges = [
                ('0h','2h'),
                ('2h','4h'),
                ('4h','6h'),
                ('6h','8h'),
                ('8h','10h')
            ]

        elif name == "welltempseq_benchmark":
            self.obs = 'timepoint'
            self.cluster_edges = [
                ('0','1'),
                ('1','2'),
                ('2','3'),
            ]

        elif name == "scntseq_benchmark":
            self.obs = 'KCl_time'
            self.cluster_edges = [
                (0,15),
                (15,30),
                (30,60),
                (60,120),
            ]
        else:
            raise ValueError("BenchMarkData name error!")

## functions used in preparing data for benchmarking

def project_to_pca(adata):
    X = adata.layers['total']
    V = adata.layers['velocity']

    X = np.array(X.A if issparse(X) else X)
    V = np.array(V.A if issparse(V) else V)
    V = np.nan_to_num(V, nan=0, neginf=0, posinf=0)
    Y = np.clip(X + V, 0, 1000)


    Xlog = np.log1p(X)
    pca = PCA()
    Xpca = pca.fit_transform(Xlog)

    Ylog = np.log1p(Y)
    Ypca = pca.transform(Ylog)
    V = Ypca - Xpca
    return V

def prepare_for_test(
    adata,
    name,
    func,
    ndims=50,
    pt=True,
    scifate2=True
):
    if scifate2:
        x_pca = adata.obsm['X_pca']
        velocity = func(adata, name)

        test = ann.AnnData(X=adata.X, obs=adata.obs, var=adata.var,
                           layers={'total':adata.layers['total'],
                                   'velocity':velocity})

        test.obsm['X_pca'] = x_pca[:,:ndims]
        test.obsm['cellrank_baseline'] = adata.obsm['velocity_cr_pca'][:,:ndims]
        if pt:
            test.obsm['pseudotime_baseline'] = adata.obsm['velocity_pst'][:,:ndims]
        else:
            ## this is a lazy implementation, will create meaningless comparison
            ## but it will never get saved
            ## this is just for the maxi dataset that we don't have a good
            ## pseudotime trajectory skeleton for.
            test.obsm['pseudotime_baseline'] = np.zeros_like(test.obsm['cellrank_baseline'])

        test.obsm['velocity_pca'] = project_to_pca(test)[:,:ndims]

        scv.pp.neighbors(test)
        return test

    else:
        x_pca = adata.obsm['X_pca']
        velocity = func(adata, name)

        test = ann.AnnData(X=adata.X, obs=adata.obs, var=adata.var,
                           layers={'total':adata.layers['total'],
                                   'velocity':velocity})

        test.obsm['X_pca'] = x_pca[:,:ndims]
        test.obsm['velocity_pca'] = project_to_pca(test)[:,:ndims]
        
        scv.pp.neighbors(test)    
        return test

def baseline_scores(
    adata
):
    X = adata.obsm['velocity_pca']
    Y1 = adata.obsm['cellrank_baseline']
    Y2 = adata.obsm['pseudotime_baseline']
    cr_scores = np.diagonal(cosine_similarity(X, Y1))
    pt_scores = np.diagonal(cosine_similarity(X, Y2))
    return cr_scores, pt_scores

def run_tests(bm):
    cbd = cross_boundary_correctness(
        bm.adata,
        k_cluster=bm.obs,
        k_velocity='velocity',
        x_emb='X_pca',
        cluster_edges=bm.cluster_edges
    )[1]

    icc = inner_cluster_coh(
        bm.adata,
        k_cluster=bm.obs,
        k_velocity='velocity',
    )[1]
    
    crs, pts = baseline_scores(bm.adata)
    
    return cbd, icc, crs, pts

def run_cbd_test_only(bm):
    cbd = cross_boundary_correctness(
        bm.adata,
        k_cluster=bm.obs,
        k_velocity='velocity',
        x_emb='X_pca',
        cluster_edges=bm.cluster_edges
    )[1]
    
    return cbd

def perform_benchmark(
    pipeline_name,
    velocity_pipeline, 
    output_folder
):
    dataset = ['mini_V3', 'mini_MN', 'mini_MD',
               'midi_NM', 'midi_Ne', 'maxi']
    
    for ds in tqdm(dataset):  
        bm_data = BenchMarkingData(ds, velocity_pipeline, pt=(ds!='maxi'))
        print(ds)
        cbd, icc, crs, pts = run_tests(bm_data)
        np.save(f'{output_folder}/{ds}_{pipeline_name}_CBD.npy', cbd)
        np.save(f'{output_folder}/{ds}_{pipeline_name}_ICC.npy', icc)
        np.save(f'{output_folder}/{ds}_{pipeline_name}_CRS.npy', crs)
        if ds!='maxi':
            np.save(f'{output_folder}/{ds}_{pipeline_name}_PTS.npy', pts)


def perform_benchmark_splicing(
    pipeline_name,
    velocity_pipeline, 
    output_folder
):
    dataset = ['splicing_mini_V3', 'splicing_mini_MN', 'splicing_mini_MD',
               'splicing_midi_NM', 'splicing_midi_Ne', 'splicing_maxi']

    
    for ds in tqdm(dataset):  
        bm_data = BenchMarkingData(ds, velocity_pipeline, pt=(ds!='splicing_maxi'))
        print(ds)
        cbd, icc, crs, pts = run_tests(bm_data)
        np.save(f'{output_folder}/{ds}_{pipeline_name}_CBD.npy', cbd)
        np.save(f'{output_folder}/{ds}_{pipeline_name}_ICC.npy', icc)
        np.save(f'{output_folder}/{ds}_{pipeline_name}_CRS.npy', crs)
        if ds!='splicing_maxi':
            np.save(f'{output_folder}/{ds}_{pipeline_name}_PTS.npy', pts)
            
def perform_benchmark_other_labelling(
    pipeline_name,
    velocity_pipeline,
    output_folder,
):
    dataset = [
        'scifate_benchmark',
        'scntseq_benchmark',
        'welltempseq_benchmark'
    ]
    
    for ds in tqdm(dataset):
        bm_data = BenchMarkingData(ds, velocity_pipeline, scifate2=False)
        print(ds)
        cbd = run_cbd_test_only(bm_data)
        np.save(f'{output_folder}/{ds}_{pipeline_name}_CBD.npy', cbd)
        
def perform_benchmark_other_splicing(
    pipeline_name,
    velocity_pipeline,
    output_folder,
):
    dataset = [
        'pancreas',
        'dentategyrus',
        'forebrain',
        'dentategyrus_lamanno',
        'gastrulation_erythroid',
        'mouse_motor_neuron'
    ]
    
    for ds in tqdm(dataset):
        bm_data = BenchMarkingData(ds, velocity_pipeline, scifate2=False)
        print()
        print(ds)
        print()
        cbd = run_cbd_test_only(bm_data)
        np.save(f'{output_folder}/{ds}_{pipeline_name}_CBD.npy', cbd)
    
def gene_specific_benchmark(output_folder, pipeline, pipeline_name):
    gene_results = {
        'mini_MN':[
            ['leiden','4','Olig2','-'],
            ['leiden','4','Tubb3','+'],
            ['leiden','2','Neurog2','+'],
            ['leiden','3','Isl2','+'],
        ],
        'mini_V3':[
            ['leiden','1','Sim1','+'],
            ['leiden','1','Sox2','-'],
            ['leiden','3','Tubb3','+'],
            ['leiden','1','Map2','+'],
        ],
        'mini_MD':[
            ['leiden','3','Sox2','-'],
            ['leiden','3','Nkx1-2','-'],
            ['leiden','3','T','-'],
            ['leiden','2','Meox1','+'],
        ],
        'midi_NM':[
            ['cell_annotation','Neural','Olig2','+'],
            ['cell_annotation','Neural','T','-'],
            ['cell_annotation','Mesoderm','Meox1','+'],
            ['cell_annotation','Early_Neural','Irx3','+'],
        ],
        'midi_Ne':[
            ['cell_annotation','Neural','Olig2','+'],
            ['cell_annotation','FP','Shh','+'],
            ['cell_annotation','P3','Nkx2-2','+'],
            ['cell_annotation','pMN','Irx3','-'],
        ]
    }
    
    scores = []
    for name, settings in gene_results.items():
        print(f"GENE SCORE: {name}")
        adata = sc.read_h5ad(f'../../data/benchmarking/{name}.h5ad')
        adata.layers['velocity'] = pipeline(adata, name)

        for seti in settings:
            sub = adata[adata.obs[seti[0]]==seti[1]]
            vel = sub[:,seti[2]].layers['velocity'].flatten()
            if seti[3]=='-':
                score = np.mean(vel<0)
            elif seti[3]=='+':
                score = np.mean(vel>0)
            scores.append(score)
           
    scores = np.array(scores)
    np.save(f'{output_folder}/{pipeline_name}_gene_specific_scores.npy', scores)
    
def gene_specific_benchmark_splicing(output_folder, pipeline, pipeline_name):
    gene_results = {
        'splicing_mini_MN':[
            ['leiden','4','Olig2','-'],
            ['leiden','4','Tubb3','+'],
            ['leiden','2','Neurog2','+'],
            ['leiden','3','Isl2','+'],
        ],
        'splicing_mini_V3':[
            ['leiden','1','Sim1','+'],
            ['leiden','1','Sox2','-'],
            ['leiden','3','Tubb3','+'],
            ['leiden','1','Map2','+'],
        ],
        'splicing_mini_MD':[
            ['leiden','3','Sox2','-'],
            ['leiden','3','Nkx1-2','-'],
            ['leiden','3','T','-'],
            ['leiden','2','Meox1','+'],
        ],
        'splicing_midi_NM':[
            ['cell_annotation','Neural','Olig2','+'],
            ['cell_annotation','Neural','T','-'],
            ['cell_annotation','Mesoderm','Meox1','+'],
            ['cell_annotation','Early_Neural','Irx3','+'],
        ],
        'splicing_midi_Ne':[
            ['cell_annotation','Neural','Olig2','+'],
            ['cell_annotation','FP','Shh','+'],
            ['cell_annotation','P3','Nkx2-2','+'],
            ['cell_annotation','pMN','Irx3','-'],
        ]
    }
    
    scores = []
    for name, settings in gene_results.items():
        print(f"GENE SCORE: {name}")
        adata = sc.read_h5ad(f'../../data/benchmarking/{name}.h5ad')
        adata.layers['velocity'] = pipeline(adata, name)

        for seti in settings:
            sub = adata[adata.obs[seti[0]]==seti[1]]
            vel = sub[:,seti[2]].layers['velocity'].flatten()
            if seti[3]=='-':
                score = np.mean(vel<0)
            elif seti[3]=='+':
                score = np.mean(vel>0)
            scores.append(score)
           
    scores = np.array(scores)
    np.save(f'{output_folder}/{pipeline_name}_gene_specific_scores.npy', scores)
    