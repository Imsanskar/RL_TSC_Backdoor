
import json

d = json.load(open('data/raw_road_net_data/gudang_5734/roadnet_4X4.json'))
# Try common JSON layouts

roads = d.get('roads') or d.get('roadnetwork', {}).get('edges') or d.get('static', {}).get('edges')

print(f'Total roads: {len(roads)}')

for r in roads[:5]:

    print(' ', r.get('id'), 'nLane:', r.get('nLane'), 'lanes_listed:', len(r.get('lanes', [])) if isinstance(r.get('lanes'), list) else 'n/a')


import xml.etree.ElementTree as ET

p = 'sumo_config/data/cityflow1x1/roadnet.net.xml'
tree = ET.parse(p)
root = tree.getroot()

# Only "real" edges (skip internal junction edges which start with ":")
edges = [e for e in root.findall('edge') if not e.get('id', '').startswith(':')]

print(f'Total real edges: {len(edges)}')
for e in edges[:8]:
    lanes = e.findall('lane')
    print(e.get('id'), 'lanes:', len(lanes))
