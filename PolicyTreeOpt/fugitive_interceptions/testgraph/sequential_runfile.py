import sys

# sys.path.append('../..')

# import pygraphviz as pgv
import networkx as nx
import pickle
import logging
import random

import pandas as pd

from ptreeopt.fugitive_interceptions.interception_model_interceptattarget import FugitiveInterception
from ptreeopt.fugitive_interceptions.visualization import plot_result

from ptreeopt.ptreeopt import PTreeOpt
from ptreeopt.ptreeopt.plotting import *


def escape_route(graph, fugitive_start, T):
    walk = []
    node = fugitive_start
    walk.append(node)

    for i in range(T - 1):
        list_neighbor = list(graph.neighbors(node))

        if i == 0:
            previous_node = node
            nextnode = random.choice(list_neighbor)

        else:
            # exclude previous node for 'normal' walk
            # don't do this for dead ends
            if len(list_neighbor) > 1:
                if previous_node in list_neighbor:
                    list_neighbor.remove(previous_node)

            # save previous node
            previous_node = node
            nextnode = random.choice(list_neighbor)

        walk.append(nextnode)
        node = nextnode

    return walk


def test_graph_2_func():
    G = nx.Graph()
    G.add_nodes_from([(0, 1), (1, 2), (2, 2), (3, 2), (4, 2), (1, 0), (2, 0), (3, 0), (4, 0), (5, 1)])
    G.add_edges_from([((0, 1), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (3, 2)), ((3, 2), (4, 2)),
                      ((0, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (4, 0)),
                      ((4, 2), (5, 1)), ((4, 0), (5, 1)),
                      ((3, 2), (3, 0)), ((3, 0), (3, 2))])
    pos = {node: node for _, node in enumerate(G.nodes())}
    labels = {node: i for i, node in enumerate(G.nodes())}
    return G, labels, pos


def test_graph_3_func():
    G = nx.Graph()
    G.add_nodes_from([(0, 1),
                      (1, 2), (2, 2), (3, 2), (4, 2),
                      (1, 1), (2, 1), (3, 1), (4, 1),
                      (1, 0), (2, 0), (3, 0), (4, 0),
                      (5, 1)])
    G.add_edges_from([((0, 1), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (3, 2)), ((3, 2), (4, 2)),
                      ((0, 1), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (3, 1)), ((3, 1), (4, 1)),
                      ((0, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (4, 0)),
                      ((4, 2), (5, 1)), ((4, 1), (5, 1)), ((4, 0), (5, 1)),
                      ((3, 2), (3, 1)), ((3, 1), (3, 2)), ((3, 0), (3, 1)), ((3, 1), (3, 0))])

    pos = {node: node for _, node in enumerate(G.nodes())}
    labels = {node: i for i, node in enumerate(G.nodes())}
    return G, labels, pos


if __name__ == '__main__':
    N = 5
    T = 5
    U = 1
    R = 9
    num_sensors = 1
    seed = 113
    multiobj = False

    np.random.seed(112)
    random.seed(112)

    graph, labels, pos = test_graph_3_func()
    units_start = [(5, 1)]
    fugitive_start = (0, 1)
    sensor_locations = [(2, 2)]
    # sensor_locations = [(0, 1)]

    # simulate fugitive escape routes
    # fugitive_routes_db = []
    # for r in range(R):
    #     route = escape_route(graph, fugitive_start, T)
    #     fugitive_routes_db.append(route)

    fugitive_routes_db = ([[(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)]] * 3 +
                          [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]] * 3 +
                          [[(0, 1), (1, 2), (2, 2), (3, 2), (4, 2)]] * 3)

    model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start,
                                 fugitive_start=fugitive_start, fugitive_routes_db=fugitive_routes_db,
                                 num_sensors=num_sensors, sensor_locations=sensor_locations, seed=seed)

    algorithm = PTreeOpt(model.f,
                         feature_bounds=[[0, T]] + [[0, 1]] * num_sensors,  # indicators
                         feature_names=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],  # indicator names
                         discrete_features=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],
                         discrete_actions=True,
                         action_names=[f"unit{u}_to_node{i}" for u in range(U) for i, _ in enumerate(graph.nodes)],
                         mu=20,  # 20
                         cx_prob=0.70,
                         population_size=100,
                         max_depth=1  # 1 when [ 0 0 1 1 1 ] (sensor stays flipped) / default 5
                         , multiobj=multiobj
                         )


    logging.basicConfig(level=logging.INFO,
                        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

    nfe_of_convergence = []
    for seed in range(100):

        best_solution, best_score, snapshots = algorithm.run(max_nfe=2000,
                                                             log_frequency=100,
                                                             snapshot_frequency=100)

        convergence_df = pd.DataFrame(snapshots)
        nfe_of_convergence.append(convergence_df[convergence_df.best_f <= 0.5].iloc[0]['nfe'])

        # best solution is not outputted AND logging frequency is strange - look into opt.py
        if multiobj:
            best_solution = best_solution[0]
        pickle.dump(snapshots, open('results/snapshots.pkl', 'wb'))

        # P = snapshots['best_P'][-1]  #best policy tree
        # colors = {f"unit{u}_to_node{i}": 'lightgrey' for u in range(U) for i, _ in enumerate(graph.nodes)}
        # graphviz_export(best_solution, 'figs/optimaltree.png', colordict=colors)  # creates one SVG

        # model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start,
        #                              fugitive_start=fugitive_start, fugitive_routes_db=fugitive_routes_db,
        #                              num_sensors=num_sensors, sensor_locations=sensor_locations, seed=seed)

        # results_df, success = model.f(best_solution, mode='simulation')
        # print('Simulation: interception succeeded: ', success)
        # print('Simulation: num intercepted: ', sum(success.values()))
        # print(results_df['policy'])

        # plot_result(graph, pos, T, U, R, results_df, success, sensor_locations, labels)

    print('avg nfe to solution: ', np.mean(nfe_of_convergence))
