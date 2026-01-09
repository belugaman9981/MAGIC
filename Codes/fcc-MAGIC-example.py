import numpy                  as np
import pandas                 as pd
import matplotlib.pyplot      as plt
from   sklearn.preprocessing  import StandardScaler
from   imblearn.over_sampling import RandomOverSampler


cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
        "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

df = pd.read_csv("magic04.data", names= cols)
df.head()
df["class"] = (df["class"] == "g").astype(int)

print(df)

for label in cols[:-1]:
        plt.hist(df[df["class"] == 1][label], color= 'blue', alpha=0.7, label='gamma', density= True)
        plt.hist(df[df["class"] == 0][label], color= 'red', alpha=0.7, label='hadron', density= True)
        plt.title(label)
        plt.ylabel("Probability")
        plt.xlabel(label)
        plt.legend()
        plt.show()
        
# Well

"""learning stuff"""

train, valid, test = np.split(df.sample(frac= 1), [int(.6 * len(df)), int(.8 * len(df))])

def scale_dataset(data, oversample= False):
    x = data[data.columns[:-1]].values
    y = data[data.columns[-1]].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data

train, X_train, y_train = scale_dataset(train, oversample= True)
valid, X_valid, y_valid = scale_dataset(valid, oversample= False)
test, X_test, y_test    = scale_dataset(test, oversample= False)

# Learning about pork aka K-Nearest Neighbors


# kNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics   import classification_report, confusion_matrix

knn_model = KNeighborsClassifier(n_neighbors= 5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print(classification_report(y_test, y_pred))
