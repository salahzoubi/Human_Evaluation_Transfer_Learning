#!/bin/bash
#SBATCH -c 24  # Number of Cores per Task
#SBATCH --mem=35G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2  # Number of GPUs
#SBATCH -t 16:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID


source activate /work/salzubi_umass_edu/.conda/envs/wmt_env
python3 /work/salzubi_umass_edu/T5_human_eval_finetune/eval_script.py --train_bsz=2 --eval_bsz=16 --num_train_steps=3000000 --es_patience=40 --val_check_interval=1 --trg_task="tgoyal_news"
