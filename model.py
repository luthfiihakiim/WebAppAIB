import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

dataset = pd.read_csv('kc_house_data.csv')

dropped_column = np.r_[0:2, 8:10, 14:17, 19:21]
train_dataframe = dataset.drop(dataset.columns[dropped_column], axis=1)

x = train_dataframe.iloc[:, 1:].values
y = train_dataframe.iloc[:, 0].values

rf_regressor = RandomForestRegressor(n_estimators=28, random_state=0)
rf_regressor.fit(x, y)

pickle.dump(rf_regressor, open('model.pkl', 'wb'))