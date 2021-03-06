  
# HUMAN VS BOT DIFFERENTIATION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Binary Classification, Machine Learning, TensorFlow

import os
os.system("cd ..")

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = 'data/Dataset.csv'
MODEL_PATH = './model/botModel'
ACC_THRESHOLD = 0.99

# DATA SEGMENTATION
data = pd.read_csv(DATASET_PATH)
print(data.describe())

x = data.iloc[:,[4,5,6,7]].values
y = data.iloc[:,8].values

# SPLITTING THE DATA INTO TRAIN AND TEST SET
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# NEURAL NETWORK FUNCTION
def botModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, input_dim = 4 , activation = 'relu'))
    model.add(tf.keras.layers.Dense(10, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    return model

# CALLBACK CLASS
class Callback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > ACC_THRESHOLD):   
            print("Reached Threshold Accuracy, Stopping Training.")   
            self.model.stop_training = True

model = botModel()
callback = Callback()

# TRAINING THE MODEL
model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
history = model.fit(x_train,y_train , epochs = 10, batch_size = 32, validation_data=(x_val,y_val), callbacks = [callback])

# PLOTTING THE GRAPH FOR TRAIN-LOSS AND VALIDATION-LOSS
plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
plt.show()
plt.savefig('graphs/loss_graph.png')

# PLOTTING THE GRAPH FOR TRAIN-ACCURACY AND VALIDATION-ACCURACY
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
plt.show()
plt.savefig('graphs/acc_graph.png')

# CALCULATING THE ACCURACY
score = model.evaluate(x_val, y_val)
print(f"Model Accuracy: {round(score[1]*100,4)}")

# SAVING THE MODEL
tf.keras.models.save_model(model,MODEL_PATH)
print(f"Successfully stored the trained model at {MODEL_PATH}")
