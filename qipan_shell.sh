# Step - victim cityflow7x28

# These commands can run simultaneously
python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc --network cityflow7x28 \
    --seed 4 --device cuda:1 --thread 8 --ngpu 1

python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc --network cityflow4x4_5816 \
    --seed 4 --device cuda:1 --thread 8 --ngpu 1

# after the previous commands finishes, then can do the fine tune commands
# Step 2 - fine-tune 

python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_rl_adversarial --network cityflow7x28 \
    --attacker_source_network cityflow4x4_5816 \
    --controller_source_network cityflow7x28 \
    --seed 4 --device cuda:1 --thread 8 --ngpu 1 --wandb

# Step 3 - test
python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_test_rl_adversarial --network cityflow7x28 \
    --attacker_source_network cityflow4x4_5816 \
    --controller_source_network cityflow7x28 \
    --seed 4 --device cuda:1 --thread 8 --ngpu 1 --wandb
