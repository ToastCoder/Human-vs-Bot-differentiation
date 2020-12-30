# HUMAN VS BOT DIFFERENTIATION

# DEVELOPED BY:
# VIGNESHWAR RAVICHANDAR

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# DATA SEGMENTATION
data = pd.read_csv('data/Dataset.csv')
x = data.iloc[:,[3,4,5]].values
y = data.iloc[:,6].values
