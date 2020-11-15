# HUMAN VS BOT DIFFERENTIATION USING K-MEANS

# DEVELOPED BY:
# VIGNESHWAR RAVICHANDAR

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# DATA SEGMENTATION
data = pd.read_csv('data/Dataset.csv')
x = data.iloc[:,[2,3,4,5]].values

# USING ELBOW METHOD TO FIND OPTIMAL NUMBER OF CLUSTERS
data_wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i,init = 'k-means++',random_state = 42)
    km.fit(x)
    data_wcss.append(km.inertia_)

# VISUALIZING THE ELBOW METHOD GRAPH
plt.plot(range(1,11),data_wcss)
plt.show()




