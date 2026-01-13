MAX_JOBS=4
total_count=0

# rm -rf /data/srg/samgain/checkpoints_hamock/test/hamock
job_count=0
mkdir -p logs
run_experiment() {
    agent=$1
    network=$2
    
    python run.py \
        --agent $agent \
        --task tsc \
        --network $network \
        --thread 8 \
        --ngpu 1 \
        --device 0 \
        --seed 1 \
        --wandb
}



for agent in g2p_colight g2p_mplight efficient_colight efficient_mplight advance_colight advance_mplight; do
    for network in cityflow4x4_5734 cityflow4x4_5816 cityflow4x4 cityflow7x28 jinan_2000 jinan_2500 jinan syn96_1; do
        run_experiment "$agent" "$network" &
        
        ((job_count++))
        ((total_count++))
        if (( job_count >= MAX_JOBS)); then
            wait -n  # Wait for any one job to finish
            ((job_count--))
        fi
        
        sleep 5
    done
done

wait  # Wait for all remaining jobs
