import networkx as nx
import pandas as pd
import numpy as np
import random
import time
import pickle
from collections import defaultdict

from ema_workbench import Model, MultiprocessingEvaluator, SequentialEvaluator, RealParameter, Constant, ScalarOutcome
from ema_workbench.em_framework import ArchiveLogger
from ema_workbench.em_framework.optimization import SingleObjectiveBorgWithArchive

# from network import test_graph_long as test_graph
from network import manhattan_graph as graph_func
from recalculate_fug_routes import recalc_fug_routes_city
from unit_ranges import unit_ranges
from sort_and_filter import sort_and_filter_pol_fug as sort_and_filter_nodes
from multiprocessing import Process, Pool, cpu_count


def run(graph_type, city, n_realizations, num_units, num_sensors, instance):
    filepath = f"../data/city/graphs/{city}.graph.graphml"
    t_max = 1800

    graph = nx.read_graphml(path=filepath, node_type=int, edge_key_type=float)

    for u, v in graph.edges():
        for i in graph[u][v]:  # if multiple edges between nodes u and v
            graph[u][v][i]['travel_time'] = float(graph[u][v][i]['travel_time'])

    labels = {}
    labels_inv = {}
    for i, node in enumerate(graph.nodes()):
        labels[node] = i
        labels_inv[i] = node

    police_start = pd.read_pickle(
        f"../data/{graph_type}/sp_const_units_start_{city}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_start = pd.read_pickle(
        f"../data/{graph_type}/sp_const_fugitive_start_{city}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes = pd.read_pickle(
        f"../data/{graph_type}/sp_const_fugitive_routes_{city}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    # fugitive_routes = [list(fugitive_routes[i].values()) for i in range(len(fugitive_routes))]  # to undo timestamps of routes

    sensor_locations = pd.read_pickle(
        f"../data/{graph_type}/sp_const_sensors_{city}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    sensor_locations_labeled = [labels[sensor] for sensor in sensor_locations]

    # fugitive_routes_labeled = []
    # for realization in range(len(fugitive_routes)):
    #     fugitive_routes_labeled.append({0: 0})
    #     for t, node in fugitive_routes[realization].items():
    #         fugitive_routes_labeled[realization][t] = labels[node]
    # fugitive_routes_labeled = np.array(fugitive_routes_labeled)
    fugitive_routes_orig = fugitive_routes.copy()

    counts = dict()
    for rep in range(len(fugitive_routes_orig)):
        counter = 0
        ground_truth_route = fugitive_routes_orig[rep]
        fugitive_routes_opt = fugitive_routes_orig.copy()
        for time_step in range((int(t_max / 30) + 1)):
            if time_step == 0:  # time_step == 0
                time_interval = [0, 0]
            elif time_step != 0:
                time_interval = [(time_step - 1) * 30, time_step * 30]

            # check sensor detection of 'ground truth' fugitive route
            sensor_triggered = [sensor in [node for t, node in ground_truth_route.items()
                                           if time_interval[0] <= t <= time_interval[1]]
                                for sensor in sensor_locations]

            if any(sensor_triggered):
                pass

            fugitive_routes_labeled_prev = fugitive_routes_opt.copy()
            # filter routes based on sensor info
            fugitive_routes_opt = recalc_fug_routes_city(fugitive_routes_opt, sensor_locations,
                                                             sensor_triggered, time_interval)

            if len(fugitive_routes_labeled_prev) > len(fugitive_routes_opt):
                counter+=1

        counts[rep] = counter

    return counts


if __name__ == '__main__':
    n_realizations = 100
    graph_type = 'city'

    for city in ['Manhattan', 'Utrecht', 'Winterswijk']:
        results = dict()
        pcts = dict()
        for num_units in [3, 10]:
            for num_sensors in [3, 10]:
                for instance in range(10):
                    counts = run(graph_type, city, n_realizations, num_units, num_sensors, instance)
                    num_reopt = sum(counts.values()) / len(counts)

                    # print(instance, num_reopt)
                    pcts[instance] = num_reopt

                results[(num_units, num_sensors)] = sum(pcts.values()) / len(pcts)

        print(city, results)
        pickle.dump(results, open(
            f'./results/num_reopt_{city}.pkl',
            'wb'))


