#! /bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=basic_slurm_exp9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00

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

source ../rl-env/bin/activate 

#./run2.sh

#./exp6.sh 

#./exp7.sh

#./zeroshot.sh

#./exp3.sh

#./exp4.sh

#./exp_sumo2.sh

#./cityflow728.sh

./exp9.sh
'''
python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_rl_adversarial --network cityflow4x4_5816 \
    --device cuda:1 --thread 8 --ngpu 1 --seed 2 --wandb


python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_rl_adversarial --network cityflow4x4_5734 \
    --device cuda:1 --thread 8 --ngpu 1 --seed 2 --wandb



python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_rl_adversarial --network cityflow1x1 \
    --attacker_source_network cityflow4x4_5816 \
    --controller_source_network cityflow1x1 \
    --seed 2 \
    --device cuda:1 --thread 8 --ngpu 1 --wandb



python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_test_rl_adversarial --network cityflow1x1 \
    --seed 2 \
    --device cuda:1 --thread 8 --ngpu 1 --wandb




python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_rl_adversarial --network cityflow1x1 \
    --attacker_source_network cityflow4x4_5734 \
    --controller_source_network cityflow1x1 \
    --seed 2 \
    --device cuda:1 --thread 8 --ngpu 1 --wandb



python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_test_rl_adversarial --network cityflow1x1 \
    --seed 2 \
    --device cuda:1 --thread 8 --ngpu 1 --wandb



'''
