# Step - victim 4x4_5734


python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc --network cityflow4x4 \
    --seed 4 --device cuda:1 --thread 8 --ngpu 1

python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc --network cityflow4x4_5734 \
    --seed 4 --device cuda:1 --thread 8 --ngpu 1

# Step 2 - fine-tune (needs exp3 done first)

python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_rl_adversarial --network cityflow4x4_5734 \
    --attacker_source_network cityflow4x4 \
    --controller_source_network cityflow4x4_5734 \
    --seed 4 --device cuda:1 --thread 8 --ngpu 1 --wandb

# Step 3 - test
python3 run.py --agent mplight --world sumo --interface libsumo \
    --task tsc_test_rl_adversarial --network cityflow4x4_5734 \
    --attacker_source_network cityflow4x4_5734 \
    --controller_source_network cityflow4x4_5734 \
    --seed 4 --device cuda:1 --thread 8 --ngpu 1 --wandb
