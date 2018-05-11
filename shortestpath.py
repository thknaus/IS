NODELIST_FILE = './nodelist.txt'
EDGELIST_FILE = './edgelist.txt'

NODELIST = []
EDGELIST = []

# Vehicle weights
U_Bahn = 1
Taxi = 2
Bus = 3


# Node object with a integer as name
class Node(object):
    def __init__(self, x):
        self.name = x

    def __str__(self):
        return str(self.name)


# Edge object with two nodes as param and string vehicle
class Edge(object):
    def __init__(self, node1, node2, vehicle):
        self.node1 = node1
        self.node2 = node2
        self.vehicle = vehicle

    def __str__(self):
        edge = '(' + str(self.node1) + ', ' + str(self.node2) + ', ' + str(self.vehicle) + ')'
        return str(edge)


# Graph object
class Graph(object):
    def __init__(self):
        # Build graph through node and edgelist
        self.graph = {}
        for n in NODELIST:
            for e in EDGELIST:
                if n.name == e.node1:
                    if n.name in self.graph:
                        self.graph[n.name].append(e.node2)
                    else:
                        self.graph[n.name] = [e.node2]

    def __str__(self):
        return str(self.graph)


def create_nodes():
    # Create nodes from list
    with open(NODELIST_FILE) as f:
        content = f.readlines()
        nodes = [int(x.strip()) for x in content]

    for n in nodes:
        NODELIST.append(Node(n))


def create_edges():
    # Create edges from list
    with open(EDGELIST_FILE) as e:
        content = e.readlines()
        edges = [x.strip() for x in content]

    for e in edges:
        l = e.split()
        for x in NODELIST:
            if x.name == int(l[0]):
                n1 = x
        for y in NODELIST:
            if y.name == int(l[1]):
                n2 = y
        EDGELIST.append(Edge(int(l[0]), int(l[1]), l[2]))


def bfs(graph, start, end):
    print(graph)
    print(start)
    print(end)


def main():
    # Create nodes
    create_nodes()
    # Create edges
    create_edges()
    # Create graph
    g = Graph()

    bfs(g, NODELIST[0], NODELIST[5])


if __name__ == "__main__":
    main()
