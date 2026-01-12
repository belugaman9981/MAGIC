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
    X = scaler.fit_transform(x)
    
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

# talking about Naive Bayes


# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))

# talking about Logistic Regression


# Logistic Regression

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression(max_iter= 1000)
lg_model = lg_model.fit(X_train, y_train)

# learning about Support Vector Machines


# SVM

from sklearn.svm import SVC

svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

# learning about neural networks


# Neural Networks + tensorflow

import tensorflow as tf

def plot_loss(history):
    fig, (axl, ax2) = plt.subplots(1, 2, figsize= (10, 4))
    axl.plot(history.history['loss'], label='loss')
    axl.plot(history.history['val_loss'], label='val_loss')
    axl.set_xlabel('Epoch')
    axl.set_ylabel('Binary crossentropy')
    axl.grid(True)
    
    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.xlabel('Epoch')
    ax2.ylabel('Accuracy')
    ax2.grid(True)
    
    plt.show()


def train_mod(X_train, y_train, num_nodes, dropout_prob, batch_size, lr, num_epochs):
        
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation= 'relu', input_shape= (10, )),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(32, activation= 'relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation= 'sigmoid')
        
    ])

    nn_model.compile(optimizer= tf.keras.optimizers.Adam(lr), 
                    loss= 'binary_crossentropy',
                    metrics= ['accuracy']
                    )
    
    history = nn_model.fit(X_train, 
        y_train, epochs= num_epochs, batch_size= batch_size, 
        validation_split= 0.2, verbose= 0
                        )
    
    return nn_model, history
    
    
epochhhs = 100

for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [16, 32, 64]:
                model, history = train_mod(X_train, y_train, num_nodes, dropout_prob, batch_size, lr, epochhhs)
                plot_loss(history)
                plot_accuracy(history)
                


