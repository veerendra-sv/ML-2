import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

data = pd.read_csv(
    r"C:\ML 2-1\PycharmProjects\PythonProject1\placement_predict_50k Dataset (3) 1.csv"
)

X = data.drop("PlacementStatus", axis=1)
X = pd.get_dummies(X)
X = X.dropna()
X = X.sample(100, random_state=42)

methods = ["single", "complete", "average", "ward"]

for method in methods:
    Z = linkage(X, method=method, metric="euclidean")

    print(f"{method.capitalize()} Linkage Heights:", Z[:, 2])

    plt.figure(figsize=(8, 5))
    dendrogram(Z)

    plt.title(method.capitalize() + " Linkage")
    plt.xlabel("Students")
    plt.ylabel("Distance")
    plt.show()