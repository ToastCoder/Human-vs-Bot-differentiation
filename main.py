# HUMAN VS BOT DIFFERENTIATION

# DEVELOPED BY:
# VIGNESHWAR RAVICHANDAR

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# DATA SEGMENTATION
data = pd.read_csv('data/Dataset.csv')
x = data.iloc[:,[3,4,5]].values
y = data.iloc[:,6].values

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# DEFINING THE NEURAL NETWORK
model = Sequential()
model.add(Dense(8, input_dim = 3 , activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


