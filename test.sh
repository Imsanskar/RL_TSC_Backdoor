agent=g2p_colight
network=cityflow4x4

python3 run.py \
    --agent $agent \
    --task tsc \
    --network $network \
    --thread 8 \
    --ngpu 1 \
    --device 0 \
    --seed 1 \
    --comet