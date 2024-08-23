import networkx as nx
import pandas as pd
import numpy as np
import random
import time
import pickle
from collections import defaultdict

from multiprocessing import Process, Pool, cpu_count
from ema_workbench import Model, MultiprocessingEvaluator, SequentialEvaluator, RealParameter, Constant, ScalarOutcome
from ema_workbench.em_framework import ArchiveLogger
from ema_workbench.em_framework.optimization import SingleObjectiveBorgWithArchive

# from network import test_graph_long as test_graph
from network import manhattan_graph as graph_func
from recalculate_fug_routes import recalc_fug_routes_city
from unit_ranges import unit_ranges
from sort_and_filter import sort_and_filter_pol_fug_city as sort_and_filter_nodes


def FIP_func(
        graph=None,
        labels=None,
        labels_perunit_sorted_inv=None,
        police_current=None,
        route_data=None,
        t_max=None,
        **kwargs,
):
    pi_nodes = {}

    for u, value in enumerate(kwargs.values()):
        associated_node = labels_perunit_sorted_inv[u][int(np.floor(value))]
        # reken hier 1 keer de reistijd naar de associated node uit ipv die hele matrix
        travel_time_to_target = nx.shortest_path_length(graph,
                                                      source=police_current[u],
                                                      target=associated_node,
                                                      weight='travel_time',
                                                      method='bellman-ford')
        associated_node = labels[associated_node]
        pi_nodes[u] = (associated_node, travel_time_to_target)

    result = set()
    for u_idx, pi_value in pi_nodes.items():  # for each police unit
        if pi_value[0] not in route_data:
            continue
        for fugitive_time in route_data[pi_value[0]]:
            if fugitive_time[1] >= (pi_value[1]):
                result.add(fugitive_time[0])
    return [len(result)]


def route_convert(route_fugitive_labeled):
    """
    returns dict {node : [(route_idx, time_to_node), ...]
    """
    route_data = dict()
    for i_r, route in enumerate(route_fugitive_labeled):
        for time_at_node_fugitive, node_fugitive in route.items():
            if node_fugitive not in route_data:
                route_data[node_fugitive] = []
            route_data[node_fugitive].append((i_r, time_at_node_fugitive))

    return route_data


def optimize(city, graph, num_nodes, police_current, upper_bounds, num_units, num_sensors, route_data, t_max,
             labels, labels_perunit_inv_sorted):
    model = Model("FIPEMA", function=FIP_func)

    model.levers = [RealParameter(f"pi_{u}", 0, upper_bounds[u]) for u in range(num_units)]

    model.constants = model.constants = [
        Constant("route_data", route_data),
        Constant("t_max", t_max),
        # Constant("tau_uv", tau_uv),
        Constant("labels", labels),
        Constant("labels_perunit_sorted_inv", labels_perunit_inv_sorted),
        # Constant("time_step", time_step),
        Constant("graph", graph),
        Constant("police_current", police_current),
    ]

    model.outcomes = [
        ScalarOutcome("num_intercepted", kind=ScalarOutcome.MAXIMIZE)
    ]

    convergence_metrics = [
        ArchiveLogger(
            f"./results/{city}/U{num_units}/S{num_sensors}",
            [l.name for l in model.levers],
            [o.name for o in model.outcomes if o.kind != o.INFO],
            base_filename=f"archives_N_{num_nodes}_U_{num_units}.tar.gz"
        ),
    ]

    with MultiprocessingEvaluator(model) as evaluator:
    # with SequentialEvaluator(model) as evaluator:
        results = evaluator.optimize(
            algorithm=SingleObjectiveBorgWithArchive,
            nfe=10000,
            searchover="levers",
            convergence=convergence_metrics,
            convergence_freq=100
        )

    return results


def run_instance(graph_type, city, n_realizations, num_units, num_sensors, instance):

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

    # sort indices on distance to start_fugitive
    labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted, labels_full_sorted_inv = sort_and_filter_nodes(
        graph,
        fugitive_start,
        fugitive_routes,
        police_start,
        t_max)

    upper_bounds = []
    for u in range(num_units):
        if len(labels_perunit_sorted[u]) <= 1:
            upper_bounds.append(0.999)
        else:
            upper_bounds.append(len(labels_perunit_sorted[u]) - 0.001)  # different for each unit

    fugitive_routes_labeled = []
    for realization in range(len(fugitive_routes)):
        fugitive_routes_labeled.append({0: 0})
        for t, node in fugitive_routes[realization].items():
            fugitive_routes_labeled[realization][t] = labels[node]

    # fugitive_routes_dict = []
    # for route in fugitive_routes:
    #     fugitive_routes_dict.append({k*35: v for k,v in enumerate(route)})
    # fugitive_routes_labeled_dict = []
    # for route in fugitive_routes_labeled:
    #     # fugitive_routes_labeled_dict.append(dict(enumerate(route)))
    #     fugitive_routes_labeled_dict.append({k*35: v for k,v in enumerate(route)})

    route_data = route_convert(fugitive_routes_labeled)

    # run optimization
    start_opt = time.time()
    results = optimize(city=graph_type,
                       graph=graph,
                       num_nodes='testgraph3',
                       # tau_uv=tau_uv,
                       police_current=police_start,
                       upper_bounds=upper_bounds,
                       num_units=num_units,
                       num_sensors=num_sensors,
                       route_data=route_data,
                       t_max=t_max,
                       labels=labels,
                       labels_perunit_inv_sorted=labels_perunit_inv_sorted)

    result_num_intercepted = results['num_intercepted'][0]
    print(result_num_intercepted)
    # print(results)

    pct_intercepted = result_num_intercepted / len(fugitive_routes)

    return pct_intercepted


if __name__ == '__main__':
    n_realizations = 100
    graph_type = 'city'
    # city = 'Manhattan'
    # num_units = 10
    # num_sensors = 3

    for city in ['Utrecht']:
        for num_units in [3]:
            for num_sensors in [3, 10]:

                pcts = dict(dict())
                for instance in range(10):
                    for seed in range(10):
                        best_seed = 0
                        pct_intercepted = run_instance(graph_type, city, n_realizations, num_units, num_sensors, instance)

                        print(num_units, num_sensors, instance, seed, pct_intercepted)

                        pickle.dump(pcts, open(
                            f'./results/{graph_type}/sp_const_nosensorinfo_{city}_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}_seed{seed}.pkl',
                            'wb'))

                        if pct_intercepted >= best_seed:
                            best_seed = pct_intercepted
                            pcts[instance] = pct_intercepted

                    pickle.dump(pcts, open(
                        f'./results/{graph_type}/sp_const_nosensorinfo_{city}_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}_upto{instance}.pkl',
                        'wb'))

                pickle.dump(pcts, open(
                    f'./results/{graph_type}/sp_const_nosensorinfo_{city}_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}.pkl',
                    'wb'))


    for city in ['Winterswijk']:
        for num_units in [10]:
            for num_sensors in [3, 10]:

                pcts = dict(dict())
                for instance in range(10):
                    for seed in range(10):
                        best_seed = 0
                        pct_intercepted = run_instance(graph_type, city, n_realizations, num_units, num_sensors, instance)

                        print(num_units, num_sensors, instance, seed, pct_intercepted)

                        pickle.dump(pcts, open(
                            f'./results/{graph_type}/sp_const_nosensorinfo_{city}_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}_seed{seed}.pkl',
                            'wb'))

                        if pct_intercepted >= best_seed:
                            best_seed = pct_intercepted
                            pcts[instance] = pct_intercepted

                    pickle.dump(pcts, open(
                        f'./results/{graph_type}/sp_const_nosensorinfo_{city}_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}_upto{instance}.pkl',
                        'wb'))

                pickle.dump(pcts, open(
                    f'./results/{graph_type}/sp_const_nosensorinfo_{city}_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}.pkl',
                    'wb'))

