sbatch -p gpu_part run_sputnik.sh a100 0.5
sbatch -p gpu_part run_sputnik.sh a100 0.625
sbatch -p gpu_part run_sputnik.sh a100 0.75
sbatch -p gpu_part run_sputnik.sh a100 0.875

sbatch -p 3090 run_sputnik.sh 3090 0.5
sbatch -p 3090 run_sputnik.sh 3090 0.625
sbatch -p 3090 run_sputnik.sh 3090 0.75
sbatch -p 3090 run_sputnik.sh 3090 0.875

sbatch -p 4090 run_sputnik.sh 4090 0.5
sbatch -p 4090 run_sputnik.sh 4090 0.625
sbatch -p 4090 run_sputnik.sh 4090 0.75
sbatch -p 4090 run_sputnik.sh 4090 0.875

