agent=advance_mplight
network=cityflow4x4

python3 run.py \
    --agent $agent \
    --task tsc_rl_adversarial \
    --network $network \
    --thread 8 \
    --ngpu 1 \
    --device 0 \
    --seed 1