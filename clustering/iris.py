from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

plt.scatter(x['Petal Length'], x['Petal Width'], c='blue')
plt.xlabel('Petal Length', fontsize=10)
plt.ylabel('Petal Width', fontsize=10)

model = KMeans(n_clusters=3, max_iter=1000)
model.fit(x)
y_labels = model.labels_

y_kmeans = model.predict(x)

accuracy = metrics.adjusted_rand_score(iris.target, y_kmeans)
print(accuracy)