import json
import os
import glob
import subprocess

ROOT_DIR = '/data/srg/samgain/RL_ITS/g2p-tsc/'
configs_dir = glob.glob('../configs/sim/*.cfg')
# print(configs_dir)
for json_file in configs_dir:
    with open(json_file, 'r') as f:
        config = json.load(f)
    
    name = os.path.basename(json_file).split('.')[0]

    roadnet = os.path.join(ROOT_DIR, 'data', config['roadnetFile'])
    flow_file = os.path.join(ROOT_DIR, 'data', config['flowFile'])
    network = config['network']
    print(name)
    subprocess.run(['bash', 'convert.sh', name, roadnet, flow_file])
