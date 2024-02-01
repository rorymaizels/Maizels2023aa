#!/bin/bash
#SBATCH --job-name=bm_dynamo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --partition=cpu
#SBATCH --output=output-dynamo.out

ml Python/3.10.8-GCCcore-12.2.0

script=B04_dynamo_benchmarking.py

environment_path=/camp/lab/briscoej/home/users/maizelr/envs_for_svm23/dynamo_env

chmod -R 744 $environment_path
source $environment_path/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

python_path=$environment_path/bin
$python_path/python $script