import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt


cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
        "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

df = pd.read_csv("magic04.data", names= cols)
df.head()
df["class"] = (df["class"] == "g").astype(int)

print(df)

for label in cols:
        plt.hist(df[df["class"] == 1][label], color= 'blue', alpha=0.7, label='gamma', density= True)
        plt.hist(df[df["class"] == 0][label], color= 'red', alpha=0.7, label='hadron', density= True)
        
        
# test123

"""learning stuff"""