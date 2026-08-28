import numpy as np
import pandas as pd

data = np.load("./datasets/uniform/T50_m5_seed1.npz")

A = data["A"]
r = data["r"]

df = pd.DataFrame(A, columns=[f"resource_{i+1}" for i in range(A.shape[1])])
df.insert(0, "reward", r)

print(df)