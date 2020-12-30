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
x = data.iloc[:,[4,5,6,7]].values
y = data.iloc[:,8].values

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# DEFINING THE NEURAL NETWORK
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, input_dim = 4 , activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

# TRAINING THE MODEL
model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
model.fit(x_train,y_train , epochs = 10, batch_size = 32,validation_data=(x_val,y_val))

# CALCULATING THE ESTIMATED ACCURACY
score = model.evaluate(x_val, y_val)
print(f"The estimated accuracy of the model is: {round(score[1]*100,4)}")
