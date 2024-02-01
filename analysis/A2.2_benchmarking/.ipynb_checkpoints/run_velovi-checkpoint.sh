#!/bin/bash
#SBATCH --job-name=bm_velovi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=output-velovi.out

ml Python/3.10.8-GCCcore-12.2.0

module load CUDA/11.1.1-GCC-10.2.0

echo "$(which python)"
echo "$(which conda)"
echo "$(nvidia-smi)"
echo "$(nvcc --version)"

script=B06_velovi_benchmarking.py

environment_path=/camp/lab/briscoej/home/users/maizelr/envs_for_svm23/velovi_env

chmod -R 744 $environment_path
source $environment_path/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

python_path=$environment_path/bin
$python_path/python $script