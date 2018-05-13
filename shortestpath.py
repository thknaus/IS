NODELIST_FILE = './nodelist.txt'
EDGELIST_FILE = './edgelist.txt'

import sys

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

#PriorityQueue
pq = Q.PriorityQueue()

NODELIST = []
EDGELIST = []
GRAPH = {}

# Vehicle weights
UBAHN = 1
TAXI = 5
BUS = 10


# Node object with a integer as name
class Node(object):
    def __init__(self, x):
        self.name = x
        self.visited = False
        self.priority = 99

    def set_priority(self, p):
        self.priority = p

    def _str_(self):
        return str(self.name)

    def __cmp__(self, other):
        return cmp(self.priority, other.priority)


# Edge object with two nodes as param and string vehicle
class Edge(object):
    def __init__(self, node1, node2, vehicle):
        self.node1 = node1
        self.node2 = node2
        if vehicle == 'Bus':
            self.vehicle = BUS
        elif vehicle == 'U_Bahn':
            self.vehicle = UBAHN
        elif vehicle == 'Taxi':
            self.vehicle = TAXI

    def __str__(self):
        edge = '(' + str(self.node1) + ', ' + str(self.node2) + ', ' + str(self.vehicle) + ')'
        return str(edge)


def init_graph():
    for n in NODELIST:
        for e in EDGELIST:
            if n.name == e.node1.name:
                if n in GRAPH:
                    GRAPH[n].append(e.node2)
                else:
                    GRAPH[n] = [e.node2]


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
        EDGELIST.append(Edge(n1, n2, l[2]))


def bfs(graph, start, end):
    print(start.name)
    print(end.name)
    insert_start(start)
    find_path(graph, start, end)


def insert_start(start):
    pq.put(start)
    print("Visit start node: " + str(start.name) + " and add to PriorityQueue")


def find_path(graph, start, end, path=[]):
    while not pq.empty():
        u = pq.get(0)
        if u.name == end.name:
            break
        else:
            child_list = graph[u]
            print("Current node: " + str(u.name))
            for v in child_list:
                if not v.visited:
                    print("Visit child node: " + str(v.name) + " from parent node: " + str(u.name))
                    if v is end:
                        print("Found end node: " + str(v.name) + " as child of node: " + str(u.name))
                        sys.exit(0)
                    v.set_priority(get_weight(u, v))
                    v.visited = True
                    pq.put(v)


def get_weight(u, v):
    for e in EDGELIST:
        if e.node1 == u and e.node2 == v:
            return e.vehicle


def main():
    # Create nodes
    create_nodes()
    # Create edges
    create_edges()
    # Create graph
    init_graph()
    g = GRAPH

    bfs(g, NODELIST[0], NODELIST[1])


if __name__ == "__main__":
    main()
