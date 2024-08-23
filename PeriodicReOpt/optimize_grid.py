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


def FIP_func(
        graph=None,
        labels=None,
        labels_perunit_sorted_inv=None,
        police_start=None,
        route_data=None,
        time_step=None,
        **kwargs,
):
    pi_nodes = {}

    for u, value in enumerate(kwargs.values()):
        associated_node = labels_perunit_sorted_inv[u][int(np.floor(value))]
        # reken hier 1 keer de reistijd naar de associated node uit ipv die hele matrix
        travel_time_to_target = nx.shortest_path_length(graph,
                                                      source=police_start[u],
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
            if fugitive_time[1] >= (pi_value[1] + time_step):
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


def optimize(city, graph, num_nodes, police_start, upper_bounds, num_units, num_sensors, route_data, time_step,
             labels, labels_perunit_inv_sorted):
    model = Model("FIPEMA", function=FIP_func)

    model.levers = [RealParameter(f"pi_{u}", 0, upper_bounds[u]) for u in range(num_units)]

    model.constants = model.constants = [
        Constant("route_data", route_data),
        Constant("labels", labels),
        Constant("labels_perunit_sorted_inv", labels_perunit_inv_sorted),
        Constant("time_step", time_step),
        Constant("graph", graph),
        Constant("police_start", police_start),
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

    # with MultiprocessingEvaluator(model) as evaluator:
    with SequentialEvaluator(model) as evaluator:
        results = evaluator.optimize(
            algorithm=SingleObjectiveBorgWithArchive,
            nfe=10000,
            searchover="levers",
            convergence=convergence_metrics,
            convergence_freq=100
        )

    return results


def run_rep(row):
    (rep, graph, graph_type, t_max, fugitive_routes, fugitive_routes_labeled_orig, ground_truth_routes, sensor_locations_labeled,
     num_units, num_sensors, police_start, police_routes, upper_bounds, labels, labels_perunit_inv_sorted) = row

    print(rep)
    police_routes_ = dict()
    fugitive_routes_labeled = fugitive_routes_labeled_orig
    ground_truth_route = fugitive_routes[rep]
    ground_truth_route_labeled = fugitive_routes_labeled[rep]

    ground_truth_routes[rep] = ground_truth_route
    intercepted = False

    for time_step in range(len(ground_truth_route)):
        if time_step == 0:
            police_current = police_start.copy()

            for u, pol_pos in enumerate(police_current):
                police_routes[rep][u] = [pol_pos]
                police_routes_[u] = [pol_pos]

        elif time_step != 0:
            # update police locations (1st step of shortest path to previous target node)
            for u in range(num_units):
                if police_current[u] != actions[u]:
                    police_current[u] = nx.shortest_path(G=graph,
                                                         source=police_current[u],
                                                         target=actions[u],
                                                         weight='travel_time',
                                                         method='bellman-ford')[1]

            for u, pol_pos in enumerate(police_current):
                police_routes[rep][u].append(pol_pos)
                police_routes_[u].append(pol_pos)

            # check for interception
            for u, pol in enumerate(police_current):
                if ground_truth_route[time_step] == pol == actions[u]:
                    intercepted = True
                    break

        if intercepted:
            pickle.dump(police_routes_, open(
                f'./data/{graph_type}/police_routes_U{num_units}_numsensors{num_sensors}_instance{instance}_rep{rep}.pkl',
                'wb'))

            return rep, intercepted


        # check sensor detection of 'ground truth' fugitive route
        sensor_triggered = [sensor == ground_truth_route_labeled[time_step] for sensor in
                            sensor_locations_labeled]

        # filter routes based on sensor info
        fugitive_routes_labeled = recalc_fug_routes(fugitive_routes_labeled, sensor_locations_labeled,
                                                    sensor_triggered, time_step)

        # update reachability {(unit, node): time_to_node, ...}
        # tau_uv = unit_ranges(start_units=police_current, U=num_units, G=graph,
        #                      labels_full_sorted=labels)

        fugitive_routes_labeled_dicts = []
        for route in fugitive_routes_labeled:
            fugitive_routes_labeled_dicts.append(dict(enumerate(route)))
        # [list(fugitive_routes[i].values()) for i in range(len(fugitive_routes))] (other way around - for DPS and PTO)

        route_data = route_convert(fugitive_routes_labeled_dicts)

        # run optimization
        start_opt = time.time()
        results = optimize(city=graph_type,
                           graph=graph,
                           num_nodes='testgraph3',
                           police_start=police_current,
                           upper_bounds=upper_bounds,
                           num_units=num_units,
                           num_sensors=num_sensors,
                           route_data=route_data,
                           time_step=time_step,
                           labels=labels,
                           labels_perunit_inv_sorted=labels_perunit_inv_sorted)
        # print(f'optimization took {time.time() - start_opt} seconds')

        # print(results)

        # extract actions for each unit from optimization results
        actions_labeled = [int(np.floor(results[f'pi_{i}'].values[0])) for i in range(num_units)]
        actions = [labels_perunit_inv_sorted[u][i] for u, i in enumerate(actions_labeled)]

    pickle.dump(police_routes_, open(
        f'./data/{graph_type}/police_routes_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}_rep{rep}.pkl',
        'wb'))
    return rep, intercepted


def run_instance(graph_type, n_realizations, manhattan_diam, num_units, num_sensors, instance):
    t_max = int(5 + (0.5 * manhattan_diam))

    graph, labels, labels_inv, pos = graph_func(manhattan_diam)

    police_start = pd.read_pickle(
        f"../data/{graph_type}/sp_const_units_start_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_start = pd.read_pickle(
        f"../data/{graph_type}/sp_const_fugitive_start_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes = pd.read_pickle(
        f"../data/{graph_type}/sp_const_fugitive_routes_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes = [list(fugitive_routes[i].values()) for i in range(len(fugitive_routes))]  # to undo timestamps of routes

    sensor_locations = pd.read_pickle(
        f"../data/{graph_type}/sp_const_sensors_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    sensor_locations_labeled = [labels[sensor] for sensor in sensor_locations]

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
        my_list = [labels[node] for node in fugitive_routes[realization]]
        fugitive_routes_labeled.append(my_list)
    # fugitive_routes_labeled = np.array(fugitive_routes_labeled)
    fugitive_routes_labeled_orig = fugitive_routes_labeled.copy()

    # num_intercepted = 0
    police_routes = defaultdict(dict)
    ground_truth_routes = dict()

    pool = Pool(processes=cpu_count()-5)
    args = []
    # for rep in range(10):
    for rep in range(len(fugitive_routes_labeled_orig)):
        args.append((rep, graph, graph_type, t_max, fugitive_routes, fugitive_routes_labeled_orig, ground_truth_routes, sensor_locations_labeled,
                num_units, num_sensors, police_start, police_routes, upper_bounds, labels, labels_perunit_inv_sorted))

    result = pool.map(run_rep, (args))
    interception_dict = {rep: intercepted for (rep, intercepted) in result}

    pool.close()
    pool.join()

    num_intercepted = sum(interception_dict.values())
    pct_intercepted = num_intercepted / len(fugitive_routes)

    return pct_intercepted


if __name__ == '__main__':
    n_realizations = 100
    graph_type = 'grid'
    manhattan_diam = 10
    t_max = int(5 + (0.5 * manhattan_diam))

    num_units = 10
    num_sensors = 10

    pcts = dict()
    for instance in [4]:
        pct_intercepted = run_instance(graph_type, n_realizations, manhattan_diam, num_units, num_sensors, instance)

        print(instance, pct_intercepted)
        pcts[instance] = pct_intercepted

        pickle.dump(pcts, open(
            f'./results/grid/sp_const_reopt_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl',
            'wb'))

    # pickle.dump(pcts, open(
    #     f'./results/grid/sp_const_reopt_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}.pkl',
    #     'wb'))

