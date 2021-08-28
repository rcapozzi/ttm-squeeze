#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:59:04 2021

@author: rcapozzi
"""

from pca import add_features
from pca import add_labels
from pycaret.classification import *

# %%
symbol = "AAPL"
df = prepare(symbol)
df.
best_model 
df.drop(['open','high','low','close','lf_max_high','lf_min_low'],axis=1,inplace=True)
df.drop(['volume', 'BTO','STO'],axis=1,inplace=True)
df.dropna(inplace=True)

x = setup(data=df, target='target')
best_model = compare_models()
plot_model(dt, plot='feature')
predict_model(dt, data=df)
