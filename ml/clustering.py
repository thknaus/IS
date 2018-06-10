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


def closestNeighborAlgo(m_size, dataset, r):
    RANGE = r

    # Create matrix
    matrix = np.zeros(shape=(m_size+1, m_size+1), dtype=Point)
    for i in dataset:
        matrix[i[0]][i[1]] = Point(0, i[0], i[1])

    check(matrix, m_size)


def check(matrix, size):
    for y in range(size):
        for x in range(size):
            if isinstance(matrix[x][y], Point):
                setCluster(matrix[x][y])
                import pdb; pdb.set_trace()
                out = nei(matrix, x, y, 1)
                import pdb; pdb.set_trace()
                print(out)
                #goToNeighbours(matrix, matrix[x][y])


def nei(mat, row, col, radius=1):
    rows, cols = len(mat), len(mat[0])
    out = []

    for i in xrange(row - radius - 1, row + radius):
        row = []
        for j in xrange(col - radius - 1, col + radius):

            if 0 <= i < rows and 0 <= j < cols:

                if isinstance(mat[i][j], Point) and mat[i][j].cluster is 0:
                    row.append(mat[i][j])

        out.append(row)

    return out


def in_bounds(matrix, row, col):
    if row < 0 or col < 0:
        return False
    if row > len(matrix)-1 or col > len(matrix)-1:
        return False
    return True


def neighbors(matrix, radius, rowNumber, colNumber):
    out = []
    for row in range(radius):
        print(row)
        for col in range(radius):
            print(col)
            if in_bounds(matrix, rowNumber+row, colNumber+col):
                print("in_bounds")
                import pdb; pdb.set_trace()
                if isinstance(matrix[rowNumber+row][colNumber+col], Point):
                    print("isntance")
                    if matrix[rowNumber+row][colNumber+col].cluster is 0:
                        out.append([rowNumber+row][colNumber+col])
    return out


def setCluster(point):
    if point.cluster == 0:
        global CLUSTER_COUNTER
        CLUSTER_COUNTER += 1
        point.cluster = CLUSTER_COUNTER


def main():
    X = generateSampleData(2, 0, 1)

    closestNeighborAlgo(1, X, 1)

    # SPECTRAL
    spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack')
    spectral.fit(X)

    y_pred = spectral.labels_.astype(np.int)

    plot_dataset(X, y_pred)


if __name__ == "__main__":
    main()
