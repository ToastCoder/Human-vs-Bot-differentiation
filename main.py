# HUMAN VS BOT DIFFERENTIATION

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
plt.title("No of clusters vs WCSS")
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()

# DEFINING AND TRAINING THE MODEL
model = KMeans(n_clusters = 2, init = 'k-means++',random_state = 42)
model.fit(x)
y = model.predict(x)

# VISUALIZING RESULTS
plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 100, color = 'blue', label = 'Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 100, color = 'green', label = 'Cluster 2')
plt.show()


