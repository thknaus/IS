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

CLUSTER_COUNTER = 0
RANGE = 0
MATRIX_LENGTH_X = 0
MATRIX_LENGTH_Y = 0


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


class Point(object):
    def __init__(self, cluster, x, y):
        self.cluster = cluster
        self.x = x
        self.y = y

    def _str_(self):
        return str(self.cluster)

    def __getitem__(self):
        return self


def closestNeighborAlgo(m_size, dataset, r):
    global RANGE
    RANGE = r

    # Create matrix
    matrix = np.zeros(shape=(m_size, m_size), dtype=Point)

    global MATRIX_LENGTH_X
    MATRIX_LENGTH_X = m_size

    global MATRIX_LENGTH_Y
    MATRIX_LENGTH_Y = m_size

    for i in dataset:
        matrix[i[0]][i[1]] = Point(0, i[0], i[1])

    check(matrix, m_size)


def check(matrix, size):
    for y in range(size):
        for x in range(size):
            if isinstance(matrix[x][y], Point):
                setCluster(matrix[x][y])
                recursivUpdateNeighbors(matrix, x, y)
                # neighborsFunction = getNeighbors(RANGE)
                # out = neighborsFunction(x, y)
                # print(out)
                # print(x, y)
                # neighborPoints = getNeighborsPoint(matrix, out)


def recursivUpdateNeighbors(matrix, x, y):
    neighborsFunction = getNeighbors(RANGE)
    out = neighborsFunction(x, y)
    neighbors = getNeighborsPoint(matrix, out)

    print(x, y)
    print(neighbors)
    for n in neighbors:
        print(n.x, n.y)
        recursivUpdateNeighbors(matrix, n.x, n.y)


def getNeighborsPoint(matrix, neighbor_list):
    points = []
    for n in neighbor_list:
        p1 = matrix[n[0]][n[1]]
        if isinstance(p1, Point) and p1.cluster == 0:
            p1.cluster=CLUSTER_COUNTER
            points.append(p1)
    return points


def getNeighbors(r):
    return lambda x, y: [(x2, y2)
            for x2 in range(x - r - 1, x+r+1)
                for y2 in range(y - r - 1, y+r+1)
                    if (-1 < x < MATRIX_LENGTH_X and
                        -1 < y < MATRIX_LENGTH_Y and
                        (x != x2 or y != y2) and
                        (0 <= x2 < MATRIX_LENGTH_X) and
                        (0 <= y2 < MATRIX_LENGTH_Y))]


def setCluster(point):
    if point.cluster == 0:
        global CLUSTER_COUNTER
        CLUSTER_COUNTER += 1
        point.cluster = CLUSTER_COUNTER


def main():
    X = generateSampleData(2, 0, 3)

    closestNeighborAlgo(4, X, 2)

    # SPECTRAL
    spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack')
    spectral.fit(X)

    y_pred = spectral.labels_.astype(np.int)

    plot_dataset(X, y_pred)


if __name__ == "__main__":
    main()
