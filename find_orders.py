"""
用 IBOrderFinder 为 6 个数据集找最优列生成顺序，结果保存到 ib_orders.json
"""
import json, sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ib_order_finder import IBOrderFinder, compute_chain_metrics

DROP_COLS = {"loan": ["Id"]}

DATASETS = ["bird", "diabetes", "house", "income", "loan", "sick"]

results = {}
for name in DATASETS:
    print(f"\n{'='*56}")
    print(f"  {name}")
    print(f"{'='*56}")
    df = pd.read_csv(f"data/{name}/train.csv")
    for c in DROP_COLS.get(name, []):
        if c in df.columns:
            df = df.drop(columns=[c])

    finder = IBOrderFinder(n_bins=100, mi_threshold_quantile=0.25)
    finder.fit(df)
    print(finder.summary())

    order = finder.get_order()
    metrics = compute_chain_metrics(order, df)
    print(f"chain_CE={metrics['chain_ce']:.4f}  chain_MI={metrics['chain_mi']:.4f}")

    results[name] = {
        "train_order": order,
        "chain_ce": metrics["chain_ce"],
        "chain_mi": metrics["chain_mi"],
    }

with open("ib_orders.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n\n=== 汇总 ===")
for name, r in results.items():
    print(f"{name:12s}  order={r['train_order']}")
    print(f"{'':12s}  chain_CE={r['chain_ce']:.4f}  chain_MI={r['chain_mi']:.4f}")

print("\n结果已保存到 ib_orders.json")
