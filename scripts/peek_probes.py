"""Print probe accuracy summary at key token positions for each variable."""
import json
r = json.load(open('results/probes/causalToM_novis_probe_results.json'))
for v in r:
    print(f"\n=== {v} ===")
    for pos in ['-1', '-4', '-7', '-8', '-15']:
        if pos not in r[v]:
            continue
        layers = sorted(r[v][pos].keys(), key=int)
        accs = [r[v][pos][L]["test_acc"] for L in layers if r[v][pos][L] is not None]
        base = r[v][pos][layers[0]]["baseline"] if r[v][pos][layers[0]] else 0
        print(f"  pos {pos:>3}  baseline={base:.2f}  | layer-accs: {[f'{a:.2f}' for a in accs]}")
