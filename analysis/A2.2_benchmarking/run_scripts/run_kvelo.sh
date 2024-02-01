#!/bin/bash
#SBATCH --job-name=bm_kvelo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=80G
#SBATCH --partition=cpu
#SBATCH --output=output-kvelo.out

script=B07_kappavelo_benchmarking.py
environment=velvet_env1

python_path=/camp/home/maizelr/.conda2/my_envs/$environment/bin
$python_path/python $script