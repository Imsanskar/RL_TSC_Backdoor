export SUMO_HOME="/data/srg/samgain/RL_ITS/dependencies/sumo-install/"
export PATH=$PATH:$SUMO_HOME/bin
export LD_LIBRARY_PATH=/data/srg/samgain/RL_ITS/dependencies/libs:$LD_LIBRARY_PATH

agent=advance_colight
network=cityflow1x1

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python SUMO_HOME="/data/srg/samgain/RL_ITS/dependencies/sumo-install/bin" python3 run.py \
    --agent $agent \
    --world sumo \
    --interface libsumo \
    --task tsc  \
    --network $network \
    --device cuda:1 \
    --thread 8 \
    --ngpu 1 \
    --device cuda:1