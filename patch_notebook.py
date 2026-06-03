import json

path = r'C:\experiment\labquake_explorer_re\export h5\2D_Summary_Plots.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update experiments in Cell 1
cell1 = nb['cells'][1]['source']
new_cell1 = []
in_exp = False
for line in cell1:
    if line.startswith('experiments = ['):
        in_exp = True
        new_cell1.append(line)
        new_cell1.append('    {"file": os.path.join(base_dir, "t0145", "t0145.csv"), "group_val": 10, "group": "10cm"},\n')
        new_cell1.append('    {"file": os.path.join(base_dir, "t0148", "t0148.csv"), "group_val": 12.5, "group": "12.5cm"},\n')
        new_cell1.append('    {"file": os.path.join(base_dir, "t0137", "t0137.csv"), "group_val": 15, "group": "15cm"},\n')
        new_cell1.append('    {"file": os.path.join(base_dir, "t0149", "t0149.csv"), "group_val": 17.5, "group": "17.5cm"},\n')
        new_cell1.append('    # {"file": os.path.join(base_dir, "t0141", "t0141.csv"), "group_val": 20, "group": "20cm"}\n')
    elif in_exp and line.strip() == ']':
        in_exp = False
        new_cell1.append(line)
    elif not in_exp:
        new_cell1.append(line)
nb['cells'][1]['source'] = new_cell1

# 2. Inject shear_force calculation into Cell 2 (process_data)
cell2 = nb['cells'][2]['source']
new_cell2 = []
for line in cell2:
    if "if target_metric not in df.columns:" in line:
        new_cell2.append("        if target_metric == 'shear_force' and 'delta_tau' in df.columns:\n")
        new_cell2.append("            # Force [kN] = delta_tau [MPa] * pi * r[cm]^2 * 0.1\n")
        new_cell2.append("            df['shear_force'] = df['delta_tau'] * np.pi * (exp['group_val'] ** 2) * 0.1\n\n")
    new_cell2.append(line)
nb['cells'][2]['source'] = new_cell2

# 3. Make shear_force use 2nd degree polynomial (Cells 3 and 4)
for i in range(len(nb['cells'])):
    if nb['cells'][i]['cell_type'] == 'code':
        cell_src = nb['cells'][i]['source']
        for j, line in enumerate(cell_src):
            if "if target_metric == 'delta_tau'" in line:
                cell_src[j] = line.replace("target_metric == 'delta_tau'", "target_metric in ['delta_tau', 'shear_force']")

# 4. Add a new cell at the end
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# === 繪製純剪力 (Shear Force) ===\n",
        "# Shear Force [kN] = Stress Drop (delta_tau) [MPa] * pi * radius^2 * 0.1\n",
        "plot_metric('shear_force', 'Shear Force (by Sigma)', r'Shear Force [kN]')\n",
        "plot_metric_by_group('shear_force', 'Shear Force (by Radius)', r'Shear Force [kN]')\n"
    ]
}
nb['cells'].append(new_cell)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Shear force logic added and cell appended.")
