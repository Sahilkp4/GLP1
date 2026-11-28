import random
n = sum(1 for _ in open("symptoms_extracted.csv")) - 1   # row count, excluding header
keep = set(random.sample(range(1, n + 1), 500))
subset = pd.read_csv("symptoms_extracted.csv", skiprows=lambda i: i > 0 and i not in keep)