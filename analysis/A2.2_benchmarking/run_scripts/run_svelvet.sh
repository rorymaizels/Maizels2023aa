#!/bin/bash
#SBATCH --job-name=bm_svelvet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=output-svelvet.out

module load CUDA/11.1.1-GCC-10.2.0

echo "$(which python)"
echo "$(which conda)"
echo "$(nvidia-smi)"
echo "$(nvcc --version)"

script=B02_svelvet_benchmarking.py
environment=velvet_env1

python_path=/camp/home/maizelr/.conda2/my_envs/$environment/bin
$python_path/python $script