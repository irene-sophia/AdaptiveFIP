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
from recalculate_fug_routes import recalc_fug_routes
from unit_ranges import unit_ranges
from sort_and_filter import sort_and_filter_pol_fug as sort_and_filter_nodes
from multiprocessing import Process, Pool, cpu_count


def run(graph_type, n_realizations, manhattan_diam, num_units, num_sensors, instance):
    t_max = int(5 + (0.5 * manhattan_diam))

    graph, labels, labels_inv, pos = graph_func(manhattan_diam)

    police_start = pd.read_pickle(
        f"../data/{graph_type}/sp_const_units_start_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_start = pd.read_pickle(
        f"../data/{graph_type}/sp_const_fugitive_start_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes = pd.read_pickle(
        f"../data/{graph_type}/sp_const_fugitive_routes_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes = [list(fugitive_routes[i].values()) for i in range(len(fugitive_routes))]  # to undo timestamps of routes

    fugitive_routes_labeled = []
    for realization in range(len(fugitive_routes)):
        my_list = [labels[node] for node in fugitive_routes[realization]]
        fugitive_routes_labeled.append(my_list)
    # fugitive_routes_labeled = np.array(fugitive_routes_labeled)
    fugitive_routes_labeled_orig = fugitive_routes_labeled.copy()

    sensor_locations = pd.read_pickle(
        f"../data/{graph_type}/sp_const_sensors_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    sensor_locations_labeled = [labels[sensor] for sensor in sensor_locations]

    counts = dict()
    for rep in range(len(fugitive_routes_labeled_orig)):
        counter = 0
        ground_truth_route_labeled = fugitive_routes_labeled_orig[rep]
        fugitive_routes_labeled_opt = fugitive_routes_labeled_orig.copy()
        for time_step in range(len(ground_truth_route_labeled)):
            # check sensor detection of 'ground truth' fugitive route
            sensor_triggered = [sensor == ground_truth_route_labeled[time_step] for sensor in
                                sensor_locations_labeled]
            fugitive_routes_labeled_prev = fugitive_routes_labeled_opt.copy()
            # filter routes based on sensor info
            fugitive_routes_labeled_opt = recalc_fug_routes(fugitive_routes_labeled_opt, sensor_locations_labeled,
                                                        sensor_triggered, time_step)

            if len(fugitive_routes_labeled_prev) > len(fugitive_routes_labeled_opt):
                counter+=1

        counts[rep] = counter

    return counts


if __name__ == '__main__':
    n_realizations = 100
    graph_type = 'grid'
    manhattan_diam = 10
    t_max = int(5 + (0.5 * manhattan_diam))

    results = dict()
    pcts = dict()
    for num_units in [3, 10]:
        for num_sensors in [3, 10]:
            for instance in range(10):
                counts = run(graph_type, n_realizations, manhattan_diam, num_units, num_sensors, instance)
                num_reopt = sum(counts.values())/len(counts)

                # print(instance, num_reopt)
                pcts[instance] = num_reopt

            results[(num_units, num_sensors)] = sum(pcts.values()) / len(pcts)

    print(results)
    pickle.dump(results, open(
        f'./results/num_reopt_grid.pkl',
        'wb'))

