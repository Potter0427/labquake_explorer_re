import json
import re

path = r'C:\experiment\labquake_explorer_re\export h5\2D_Summary_Plots.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 1
cell1_source = nb['cells'][1]['source']
for i, line in enumerate(cell1_source):
    # Remove "marker": "..." from the line
    line = re.sub(r'\"marker\":\s*\"[^\"]+\"\s*,\s*', '', line)
    cell1_source[i] = line
nb['cells'][1]['source'] = cell1_source

# Update Cell 2
cell2_source = nb['cells'][2]['source']
for i, line in enumerate(cell2_source):
    if "'marker': exp['marker']," in line:
        cell2_source[i] = ""
nb['cells'][2]['source'] = [line for line in cell2_source if line != ""]

# Update Cell 4
cell4_source = nb['cells'][4]['source']
new_cell4_source = []
skip = False
for line in cell4_source:
    if line.startswith('SIGMA_MARKER'):
        skip = True
    if skip and '}' in line:
        skip = False
        continue
    if not skip and not line.startswith('SIGMA_MARKER'):
        new_cell4_source.append(line)
nb['cells'][4]['source'] = new_cell4_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Removed marker definitions from config.")
