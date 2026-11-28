import random
import pandas as pd

path = "symptoms_extracted.csv"

# Count data rows (excluding header)
with open(path, encoding="utf-8") as f:
    n = sum(1 for _ in f) - 1

keep = set(random.sample(range(1, n + 1), 500))

subset = pd.read_csv(path, skiprows=lambda i: i > 0 and i not in keep)
subset.to_csv("symptoms_subset_500.csv", index=False)
print(subset.shape)
