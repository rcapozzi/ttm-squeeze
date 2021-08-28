#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

from pca import add_features
from pca import add_labels
#%%
def prepare(symbol):
    df =  pd.read_csv(f"datasets/{symbol}.csv.gz", index_col=0)
    df = add_features(df)
    add_labels(df)
    return df
    
# %%
def model_trade(symbol):
    df = prepare(symbol)
    df['target'] = -1
    df.loc[ (df.BTO == 1 ) & (df.STO == 0), 'target' ] = 1
    df.loc[ (df.BTO == 0 ) & (df.STO == 1), 'target' ] = 0
    df = df.loc[(df.target > -1)].copy()
    
    df.dropna(inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, features], df.target, test_size=0.2, random_state=0)

    #setting parameters for network layers     
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=len(x_train.columns)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
     
    #setting up the model compiler
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
    model.fit(x_train, y_train, epochs=32, use_multiprocessing=True, verbose=2)

    y_hat = model.predict(x_test)
    y_hat = [0 if val < 0.5 else 1 for val in y_hat]
    model._accuracy_score = accuracy_score(y_test, y_hat)
    return model

#%%
def model_hold(symbol):
    df = prepare(symbol)
    df.dropna(inplace=True)
    labels = pd.get_dummies(df.target)
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, features], labels.hold, test_size=0.2, random_state=0)

    #setting parameters for network layers     
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=len(x_train.columns)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
     
    #setting up the model compiler
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
    model.fit(x_train, y_train, epochs=32, batch_size=32)

    y_hat = model.predict(x_test)
    y_hat = [0 if val < 0.5 else 1 for val in y_hat]
    accuracy_score(y_test, y_hat)
    model._accuracy_score = accuracy_score(y_test, y_hat)
    return model
    # model.save('tfmodel')
    # model = load_model('tfmodel')    

# %%
symbol = "AAPL"
df = prepare(symbol)
model = model_hold(symbol)
model.predict(df.loc[:,features].tail(10))

model = model_trade(symbol)
model.predict(df.loc[:,features].tail(10))

