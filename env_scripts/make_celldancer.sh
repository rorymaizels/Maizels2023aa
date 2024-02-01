#!/bin/bash
#SBATCH --job-name=cdn_env
#SBATCH --output=cdance-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1

ml Python/3.10.8-GCCcore-12.2.0

name=celldancer
environment_path=/camp/lab/briscoej/home/users/maizelr/envs_for_svm23/${name}_env

python -m venv $environment_path
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

chmod -R 744 $environment_path
source $environment_path/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi


pip_path=$(which pip)
if [[ ! $pip_path == $environment_path* ]]; then
    echo "Pip path is not within the virtual environment."
    deactivate
    exit 1
fi

pip install --upgrade pip
pip install --no-cache-dir scanpy==1.9.*
pip install --no-cache-dir scvelo==0.2.*
pip install --no-cache-dir anndata==0.8.0
pip install --no-cache-dir numpy==1.23.5
pip install --no-cache-dir matplotlib==3.5.3 
pip install --no-cache-dir pandas==1.5.3
pip install --no-cache-dir scipy==1.8.*
pip install --no-cache-dir scikit-learn==1.3.*
pip install --no-cache-dir seaborn==0.12.*
pip install --no-cache-dir tqdm
pip install --no-cache-dir ipykernel

ipython kernel install --user --name=${name}_env

deactivate

echo "All done!"