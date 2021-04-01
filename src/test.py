#-------------------------------------------------------------------------------------------------------------------------------

# HUMAN VS BOT DIFFERENTIATION

# FILE NAME: test.py

# DEVELOPED BY: Vigneshwar Ravichandar, Moulishankar M R

# TOPICS: Multiclass Classification, Machine Learning, TensorFlow

#-------------------------------------------------------------------------------------------------------------------------------

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = 'data/Dataset.csv'
MODEL_PATH = './model/botModel'

# DATA PREPROCESSING
data = pd.read_csv(DATASET_PATH)
print(data.describe())

x = data.iloc[:,[4,5,6,7]].values
y = data.iloc[:,8].values

# OPENING THE TRAINED MODEL
model = tf.keras.models.load_model(MODEL_PATH)

features = ['Check Status','Captcha Attempts','No of Login Attempts','Avg Time between Attempts']
response = []
for i in range(len(features)):
    response.append(float(input(f"Please mention the {features[i]}: ")))
res = model.predict_classes([response])
if res == 1:
    print("It might be a Human.")
elif res == 0:
    print("It might be a Bot.")
