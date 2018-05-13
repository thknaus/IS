NODELIST_FILE = './nodelist.txt'
EDGELIST_FILE = './edgelist.txt'

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
U_Bahn = 1
Taxi = 5
Bus = 10


# Node object with a integer as name
class Node(object):
    def __init__(self, x):
        self.name = x
        self.visited = False
        self.priority = 99
<<<<<<< HEAD
=======

    def get_visited():
        return self.visited

    def set_visited(v):
        self.visited = v

    def set_priority(p):
        self.priority = p
>>>>>>> 9f8d76a2f5671f18f34c75370206d5fe41359c90

    def get_visited():
        return self.visited

    def set_visited(v):
        self.visited = v

    def set_priority(p):
        self.priority = p

    def _str_(self):
        return str(self.name)

<<<<<<< HEAD
    def _cmp_(self, other):
=======
    def __cmp__(self, other):
>>>>>>> 9f8d76a2f5671f18f34c75370206d5fe41359c90
        return cmp(self.priority, other.priority)


# Edge object with two nodes as param and string vehicle
class Edge(object):
    def __init__(self, node1, node2, vehicle):
        self.node1 = node1
        self.node2 = node2
        self.vehicle = vehicle

    def __str__(self):
        edge = '(' + str(self.node1) + ', ' + str(self.node2) + ', ' + str(self.vehicle) + ')'
        return str(edge)


def init_graph():
    for n in NODELIST:
        for e in EDGELIST:
            if n.name == e.node1:
                if n.name in GRAPH:
                    GRAPH[n.name].append(e.node2)
                else:
                    GRAPH[n.name] = [e.node2]


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
    insert_start(start)
    find_path(graph, start, end)


def insert_start(start):
    pq.put(start)
    print("Insert PriorityQueue")


def find_path(graph, start, end, path=[]):
    while not pq.empty():
        u = pq.get(0)
        if u == end:
            break
        else:
            import pdb; pdb.set_trace()
<<<<<<< HEAD
            child_list = graph[u.name]
=======
            child_list = graph[u]
>>>>>>> 9f8d76a2f5671f18f34c75370206d5fe41359c90
            print(child_list)
            for v in child_list:
                if not v.get_visited:
                    v.set_priority(get_weight(u, v))
                    v.set_visited(True)
                    pq.put(v)
        break
    print("Hab's diggi")



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

    print("EdgeList")
    for y in EDGELIST:
        print(y)
    print("NodeList")
    for x in NODELIST:
        print(x)
    bfs(g, NODELIST[0], NODELIST[5])
    print("Graph")
    print(g)
<<<<<<< HEAD
=======





>>>>>>> 9f8d76a2f5671f18f34c75370206d5fe41359c90


if __name__ == "__main__":
    main()
