from mealpy_comparison_v2 import get_all_algorithms_v3

algos = get_all_algorithms_v3()
names = [a['full_name'] for a in algos]

targets = ['SSA.OriginalSSA', 'SSA.DevSSA']
for t in targets:
    print(f"{t} → {'VAR' if t in names else 'YOK'}")
