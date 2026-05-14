#! /bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=basic_slurm_job_jinan_baselineattack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"
echo ""

cd $SLURM_SUBMIT_DIR
echo "Hello World! I ran on compute node $(/bin/hostname -s)"

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"

source ../tsc-env/bin/activate 

#./run2.sh

agent=mplight
#network=cityflow7x28
#network=cityflow7x28
'''
network=jinan


python3 run.py \
    --agent $agent \
    --world sumo \
    --interface libsumo \
    --task tsc \
    --network $network \
    --device cuda:1 \
    --thread 8 \
    --ngpu 1 \
    --device cuda:0 \
    --wandb
'''

agent=mplight
network=cityflow1x1_sb_sx
'''
python3 run.py \
    --agent $agent \
    --world sumo \
    --interface libsumo \
    --task tsc \
    --network $network \
    --device cuda:1 \
    --thread 8 \
    --ngpu 1 \
    --device cuda:0 \
    --wandb



python3 run.py \
    --agent mplight \
    --world sumo \
    --interface libsumo \
    --task tsc_rl_adversarial \
    --network cityflow1x1 \
    --device cuda:1 \
    --thread 8 \
    --ngpu 1 \
    --wandb
python3 run.py \
    --agent $agent \
    --world sumo \
    --interface libsumo \
    --task tsc_rl_adversarial  \
    --controller_source_network jinan \
    --attacker_source_network jinan \
    --network $network \
    --device cuda:1 \
    --thread 8 \
    --ngpu 1 \
    --device cuda:1 \
    --wandb
'''
python3 run.py \
    --agent mplight \
    --world sumo \
    --interface libsumo \
    --task tsc_rl_adversarial \
    --network jinan \
    --device cuda:1 \
    --thread 8 \
    --ngpu 1 \
    --wandb
