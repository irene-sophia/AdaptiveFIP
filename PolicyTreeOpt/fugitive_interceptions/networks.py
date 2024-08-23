import networkx as nx


def manhattan_graph(**kwargs):
    """
    Generate manhattan graph

    Parameters
    ----------
    N : int
        width of graph.

    Returns
    -------
    G : networkx graph
        graph of size (N,N).
    labels : dict
        {pos: vertex number}.
    pos : dict
        {vertex:pos}.

    """

    N = kwargs['N']
    G = nx.grid_2d_graph(N, N)
    pos = dict((n, n) for n in G.nodes())
    #labels = dict(((i, j), i * N + j) for i, j in G.nodes())  #
    labels = {node: str(i) for i, node in enumerate(G.nodes())}

    # set travel time (unidistant)
    travel_time = 1
    nx.set_edge_attributes(G, travel_time, "travel_time")

    return G, labels, pos


def rotterdam_graph(**kwargs):
    filepath = "C:/Users/isvandroffelaa/Documents/MSc studenten/Tom/Escape_Route_Model/data/graphs/Graph_city_centre.graph.graphml"

    graph = nx.read_graphml(path=filepath
                            #node_type={"connecting_out_nodes": int, "speed_kph": float},
                            #edge_dtypes={"travel_time_q3": float, 'travel_time_q1': float,
                            #             "traffic_density": float, 'lanes': int}
                            )
    pos = {}
    for node, data in graph.nodes.data():
        pos[node] = (float(data['x']), float(data['y']))

    return graph, {}, pos
