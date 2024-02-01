#!/bin/bash
#SBATCH --job-name=bm_unitvelo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=120G
#SBATCH --partition=cpu
#SBATCH --output=output-unitvelo.out

ml Python/3.10.8-GCCcore-12.2.0

script=B05_unitvelo_benchmarking.py

environment_path=/camp/lab/briscoej/home/users/maizelr/envs_for_svm23/unitvelo_env

chmod -R 744 $environment_path
source $environment_path/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

python_path=$environment_path/bin
$python_path/python $script