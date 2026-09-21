import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from Knee_for_DBSCAN import find_eps

data=pd.read_csv("preprocessed_placement_dataset.csv")

if len(data)>1000:
    data=data.sample(1000,random_state=42)
    X=data.drop("PlacementStatus",axis=1)

    model=DBSCAN(eps=find_eps(X,min_samples=5),min_samples=5)

    labels=model.fit_predict(X)
    core=model.core_sample_indices_
    noise=labels==-1
    border=~noise & ~pd.Series(range(len(X))).isin(core)

    print("Clusters:",len(set(labels))-(1 if -1 in labels else 0))
    print("Core:",len(core))
    print("Border:",border.sum())
    print("Noise",noise.sum())

    plt.scatter(X.iloc[:,0],X.iloc[:,1],c=labels)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("DBSCAN Clustering")
    plt.show()