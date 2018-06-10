import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# import spectral_clustering as spectralClustering
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice
from random import randint


def plot_dataset(X, y_pred=[0], quantile=.3, n_neighbors=10, fname=None):
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))
    plt.figure(figsize=(10, 10))

    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    if fname:
        plt.savefig(fname)
    plt.show()


def generateSampleData(item_number, min_value, max_value):
    df = pd.DataFrame(columns=['x', 'y'])
    for x in range(0, item_number):
        df.loc[x] = [randint(min_value, max_value) for n in range(2)]

    list = []
    for i in range(0, len(df.index)):
        inner_list = []
        inner_list.append(df['x'].iloc[i])
        inner_list.append(df['y'].iloc[i])
        list.append(inner_list)
    return list


def closestNeighborAlgo(m_size, dataset, range):

    for i in range(5):
        print(i)
    # Create matrix
    matrix = np.zeros(shape=(m_size+1, m_size+1))
    for i in dataset:
        matrix[i[0]][i[1]] = 1
    size = m_size + 1




def main():
    X = generateSampleData(10, 0, 10)
    print(X)
    closestNeighborAlgo(10, X, 1)

    # SPECTRAL
    spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack')
    spectral.fit(X)

    y_pred = spectral.labels_.astype(np.int)

    plot_dataset(X, y_pred)


if __name__ == "__main__":
    main()
