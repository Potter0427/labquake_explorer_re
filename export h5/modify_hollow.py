import json
import os

file_path = r"e:\experiment\labquake_explorer_re\export h5\2D_Summary_Plots.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            if "markeredgecolor=color" in line and "markerfacecolor='none'" not in line:
                line = line.replace("markeredgecolor=color", "markeredgecolor=color, markerfacecolor='none'")
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Modified notebook successfully.")
