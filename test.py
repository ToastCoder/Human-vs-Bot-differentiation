# HUMAN VS BOT DIFFERENTIATION

# FILE NAME: test.py

# DEVELOPED BY: Vigneshwar Ravichandar, Moulishankar M R

# TOPICS: Multiclass Classification, Machine Learning, TensorFlow

# DISABLE TENSORFLOW DEBUG INFORMATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print("TensorFlow Debugging Information is hidden.")

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = 'data/custData.csv'
MODEL_PATH = './model/botModel'

# DATA PREPROCESSING
data = pd.read_csv(DATASET_PATH)
print(data.describe())

x = data.iloc[:,[4,5,6,7]].values
y = data.iloc[:,8].values

# OPENING THE TRAINED MODEL
model = tf.keras.models.load_model(MODEL_PATH)