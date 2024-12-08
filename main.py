import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Дані з Дз_1
data = {
    "Error Param": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "Mean": [20.01, 15.01, 13.34, 12.50, 12.00, 11.67, 11.42, 11.25, 11.10, 11.00],
    "Variance": [100.54, 25.06, 11.21, 6.26, 3.98, 2.79, 2.01, 1.56, 1.23, 0.99],
    "Std Dev": [10.02, 5.00, 3.35, 2.50, 1.99, 1.67, 1.42, 1.25, 1.11, 0.99],
}

# Перетворення даних у DataFrame
df = pd.DataFrame(data)

# Вибір колонок для кластеризації
X = df[["Mean", "Variance", "Std Dev"]]

# Виконання кластеризації за допомогою k-means
kmeans = KMeans(n_clusters=3, random_state=42)  # Вибір 3 кластерів для прикладу
kmeans.fit(X)

# Додавання результатів кластеризації у DataFrame
df["Cluster"] = kmeans.labels_

# Візуалізація результатів
plt.scatter(X["Mean"], X["Variance"], c=df["Cluster"], cmap="viridis")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="red",
    label="Centers",
)
plt.xlabel("Mean")
plt.ylabel("Variance")
plt.title("Кластеризація за допомогою k-means")
plt.legend()
plt.show()

# Висновки
print("Центри кластерів:", kmeans.cluster_centers_)
print("Результати кластеризації:")
print(df)
