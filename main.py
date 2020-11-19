# HUMAN VS BOT DIFFERENTIATION

# DEVELOPED BY:
# VIGNESHWAR RAVICHANDAR

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# DATA SEGMENTATION
data = pd.read_csv('data/Dataset.csv')
x = data.iloc[:,[3,4,5]].values
y = data.iloc[:,6].values

# DEFINING AND TRAINING A MODEL
model = SVC(kernel = 'rbf',random_state = 0)
model.fit(x,y)

y_pred = model.predict(x)


plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 50, c = 'red', label = 'Humans')
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 50, c = 'blue', label = 'Bots')
plt.show()

