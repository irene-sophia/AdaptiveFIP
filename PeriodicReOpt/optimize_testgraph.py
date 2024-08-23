<<<<<<< HEAD
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
from network import test_graph as test_graph
from recalculate_fug_routes import recalc_fug_routes
from unit_ranges import unit_ranges
from sort_and_filter import sort_and_filter_pol_fug as sort_and_filter_nodes


def FIP_func(
        tau_uv=None,
        labels=None,
        labels_perunit_sorted_inv=None,
        route_data=None,
        time_step=None,
        **kwargs,
):
    pi_nodes = {}

    for u, value in enumerate(kwargs.values()):
        associated_node = labels_perunit_sorted_inv[u][int(np.floor(value))]
        associated_node = labels[associated_node]
        pi_nodes[u] = (associated_node, tau_uv[u, associated_node])

    result = set()
    for u_idx, pi_value in pi_nodes.items():  # for each police unit
        if pi_value[0] not in route_data:
            continue
        for fugitive_time in route_data[pi_value[0]]:
            if fugitive_time[1] >= (pi_value[1] + time_step):
                # if fugitive_time[1] <= time_horizon  # TODO: MPC style

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


def optimize(city, vary_param, num_nodes, tau_uv, upper_bounds, num_units, route_data, time_step,
             labels, labels_perunit_inv_sorted):
    model = Model("FIPEMA", function=FIP_func)

    model.levers = [RealParameter(f"pi_{u}", 0, upper_bounds[u]) for u in range(num_units)]

    model.constants = model.constants = [
        Constant("route_data", route_data),
        Constant("tau_uv", tau_uv),
        Constant("labels", labels),
        Constant("labels_perunit_sorted_inv", labels_perunit_inv_sorted),
        Constant("time_step", time_step),
    ]

    model.outcomes = [
        ScalarOutcome("num_intercepted", kind=ScalarOutcome.MAXIMIZE)
    ]

    convergence_metrics = [
        ArchiveLogger(
            f"./results/{city}/{vary_param}",
            [l.name for l in model.levers],
            [o.name for o in model.outcomes if o.kind != o.INFO],
            base_filename=f"archives_N_{num_nodes}_U_{num_units}.tar.gz"
        ),
    ]

    # with MultiprocessingEvaluator(model) as evaluator:
    with SequentialEvaluator(model) as evaluator:
        results = evaluator.optimize(
            algorithm=SingleObjectiveBorgWithArchive,
            nfe=100,
            searchover="levers",
            convergence=convergence_metrics,
            convergence_freq=100
        )

    return results


if __name__ == '__main__':
    start_all = time.time()
    n_realizations = 10
    t_max = 5
    graph_type = 'test_graph_3'

    if graph_type == 'test_graph_2':
        graph, labels, labels_inv, pos = test_graph(n_paths=2)
    if graph_type == 'test_graph_3':
        graph, labels, labels_inv, pos = test_graph(n_paths=3)

    num_units = 1

    # police_start = pd.read_pickle(
    #     f"../data/{graph_type}/units_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")  # [(3, 0)] ???
    # police_start = [(7, 1)]
    police_start = [(5, 1)]
    fugitive_start = pd.read_pickle(
        f"../data/{graph_type}/fugitive_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")
    # fugitive_routes = pd.read_pickle(
    #     f"../data/{graph_type}/fugitive_routes_T{t_max}_R{n_realizations}_U{num_units}.pkl")

    fugitive_routes = ([[(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)]] * 333 +
                       [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]] * 333 +
                       [[(0, 1), (1, 2), (2, 2), (3, 2), (4, 2)]] * 333)
    n_realizations = len(fugitive_routes)

    # sort indices on distance to start_fugitive
    labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted, labels_full_sorted_inv = sort_and_filter_nodes(
        graph,
        fugitive_start,
        fugitive_routes,
        police_start,
        t_max)

    # sensor_locations = [(2, 2), (2, 1)]
    # sensor_locations = [(1, 0)]
    sensor_locations = [(2, 0)]
    # sensor_locations = [(2, 1)]
    # sensor_locations = [(0, 1)]
    sensor_locations_labeled = [labels[sensor] for sensor in sensor_locations]

    upper_bounds = []
    for u in range(num_units):
        if len(labels_perunit_sorted[u]) <= 1:
            upper_bounds.append(0.999)
        else:
            upper_bounds.append(len(labels_perunit_sorted[u]) - 0.001)  # different for each unit

    fugitive_routes_labeled = []
    for realization in range(len(fugitive_routes)):
        my_list = []
        for t in range(t_max):
            my_list.append(labels[fugitive_routes[realization][t]])
        fugitive_routes_labeled.append(my_list)
    fugitive_routes_labeled = np.array(fugitive_routes_labeled)
    fugitive_routes_labeled_orig = fugitive_routes_labeled.copy()

    num_intercepted = 0
    police_routes = defaultdict(dict)
    ground_truth_routes = dict()
    interception_dict = dict()

    for rep in range(len(fugitive_routes_labeled_orig)):
    # for rep in range(10):
        print(rep)
        interception_dict[rep] = 0
        fugitive_routes_labeled = fugitive_routes_labeled_orig
        ground_truth_route = fugitive_routes[rep]
        ground_truth_route_labeled = fugitive_routes_labeled[rep]

        ground_truth_routes[rep] = ground_truth_route
        intercepted = False

        # print('ground truth', ground_truth_route)

        for time_step in range(t_max):
            # print('time step: ', time_step)
            if time_step != 0:
                # update police locations (1st step of shortest path to previous target node)
                for u in range(num_units):
                    if police_current[u] != actions[u]:
                        police_current[u] = nx.shortest_path(G=graph,
                                                             source=police_current[u],
                                                             target=actions[u])[1]

                for u, pol_pos in enumerate(police_current):
                    police_routes[rep][u].append(pol_pos)

            else:  # time_step == 0
                police_current = police_start.copy()

                for u, pol_pos in enumerate(police_current):
                    police_routes[rep][u] = [pol_pos]

            # print(time_step, 'police_current: ', police_current)

            # check for interception
            for u, pol in enumerate(police_current):
                # if ground_truth_route[time_step] == pol:
                #     if pol != actions[u]:
                #         print('weird case')

                if ground_truth_route[time_step] == pol == actions[u]:
                    num_intercepted += 1
                    intercepted = True
                    interception_dict[rep] = 1

                    # print(time_step, 'intercepted!')
                    # print(time_step, 'ground truth route: ', ground_truth_route)
                    # print(time_step, 'target nodes: ', actions)
                    # print(time_step, 'police_current', police_current)
                    # print('__________________')

                    break

            if intercepted:
                break

            # check interception of 'ground truth' fugitive route
            sensor_triggered = [sensor == ground_truth_route_labeled[time_step] for sensor in
                                sensor_locations_labeled]

            # if True in sensor_triggered:
                # print(time_step, 'sensor triggered!')

            # filter routes based on sensor info
            fugitive_routes_labeled = recalc_fug_routes(fugitive_routes_labeled, sensor_locations_labeled,
                                                        sensor_triggered, time_step)

            # update reachability {(unit, node): time_to_node, ...}
            # TODO: check this func: maybe time step should be different? can the unit still adjust after the sensor in time?
            tau_uv = unit_ranges(start_units=police_current, U=num_units, G=graph,
                                 labels_full_sorted=labels)

            fugitive_routes_labeled_dicts = []
            for route in fugitive_routes_labeled:
                fugitive_routes_labeled_dicts.append(dict(enumerate(route)))
            # [list(fugitive_routes[i].values()) for i in range(len(fugitive_routes))] (other way around - for DPS and PTO)

            route_data = route_convert(fugitive_routes_labeled_dicts)

            # run optimization
            start_opt = time.time()
            results = optimize(city=graph_type,
                               vary_param='',
                               num_nodes='testgraph3',
                               tau_uv=tau_uv,
                               upper_bounds=upper_bounds,
                               num_units=num_units,
                               route_data=route_data,
                               time_step=time_step,
                               labels=labels,
                               labels_perunit_inv_sorted=labels_perunit_inv_sorted)
            # print(f'optimization took {time.time() - start_opt} seconds')

            # print(results)

            # extract actions for each unit from optimization results
            actions_labeled = [int(np.floor(results[f'pi_{i}'].values[0])) for i in range(num_units)]
            actions = [labels_perunit_inv_sorted[u][i] for u, i in enumerate(actions_labeled)]

            # print(time_step, 'actions: ', actions, 'with expected pct of remaining routes intercepted: ',
            #       (results['num_intercepted'].values[0] / len(fugitive_routes_labeled)))


    print(f'END: pct intercepted: {num_intercepted / len(fugitive_routes)}')
    print(f'END: script took {(time.time() - start_all) / 60} minutes')

    pickle.dump(police_routes,
                open(f"./results/{graph_type}/result_police_routes_{graph_type}_S{len(sensor_locations)}.pkl", "wb"))
    pickle.dump(ground_truth_routes,
                open(f"./results/{graph_type}/result_fug_routes_{graph_type}_S{len(sensor_locations)}.pkl", "wb"))
    pickle.dump(interception_dict,
                open(f"./results/{graph_type}/result_interception_{graph_type}_S{len(sensor_locations)}.pkl", "wb"))

    interception_dict

=======
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
from network import test_graph as test_graph
from recalculate_fug_routes import recalc_fug_routes
from unit_ranges import unit_ranges
from sort_and_filter import sort_and_filter_pol_fug as sort_and_filter_nodes


def FIP_func(
        tau_uv=None,
        labels=None,
        labels_perunit_sorted_inv=None,
        route_data=None,
        time_step=None,
        **kwargs,
):
    pi_nodes = {}

    for u, value in enumerate(kwargs.values()):
        associated_node = labels_perunit_sorted_inv[u][int(np.floor(value))]
        associated_node = labels[associated_node]
        pi_nodes[u] = (associated_node, tau_uv[u, associated_node])

    result = set()
    for u_idx, pi_value in pi_nodes.items():  # for each police unit
        if pi_value[0] not in route_data:
            continue
        for fugitive_time in route_data[pi_value[0]]:
            if fugitive_time[1] >= (pi_value[1] + time_step):
                # if fugitive_time[1] <= time_horizon  # TODO: MPC style

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


def optimize(city, vary_param, num_nodes, tau_uv, upper_bounds, num_units, route_data, time_step,
             labels, labels_perunit_inv_sorted):
    model = Model("FIPEMA", function=FIP_func)

    model.levers = [RealParameter(f"pi_{u}", 0, upper_bounds[u]) for u in range(num_units)]

    model.constants = model.constants = [
        Constant("route_data", route_data),
        Constant("tau_uv", tau_uv),
        Constant("labels", labels),
        Constant("labels_perunit_sorted_inv", labels_perunit_inv_sorted),
        Constant("time_step", time_step),
    ]

    model.outcomes = [
        ScalarOutcome("num_intercepted", kind=ScalarOutcome.MAXIMIZE)
    ]

    convergence_metrics = [
        ArchiveLogger(
            f"./results/{city}/{vary_param}",
            [l.name for l in model.levers],
            [o.name for o in model.outcomes if o.kind != o.INFO],
            base_filename=f"archives_N_{num_nodes}_U_{num_units}.tar.gz"
        ),
    ]

    # with MultiprocessingEvaluator(model) as evaluator:
    with SequentialEvaluator(model) as evaluator:
        results = evaluator.optimize(
            algorithm=SingleObjectiveBorgWithArchive,
            nfe=100,
            searchover="levers",
            convergence=convergence_metrics,
            convergence_freq=100
        )

    return results


if __name__ == '__main__':
    start_all = time.time()
    n_realizations = 10
    t_max = 5
    graph_type = 'test_graph_3'

    if graph_type == 'test_graph_2':
        graph, labels, labels_inv, pos = test_graph(n_paths=2)
    if graph_type == 'test_graph_3':
        graph, labels, labels_inv, pos = test_graph(n_paths=3)

    num_units = 1

    # police_start = pd.read_pickle(
    #     f"../data/{graph_type}/units_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")  # [(3, 0)] ???
    # police_start = [(7, 1)]
    police_start = [(5, 1)]
    fugitive_start = pd.read_pickle(
        f"../data/{graph_type}/fugitive_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")
    # fugitive_routes = pd.read_pickle(
    #     f"../data/{graph_type}/fugitive_routes_T{t_max}_R{n_realizations}_U{num_units}.pkl")

    fugitive_routes = ([[(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)]] * 333 +
                       [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]] * 333 +
                       [[(0, 1), (1, 2), (2, 2), (3, 2), (4, 2)]] * 333)
    n_realizations = len(fugitive_routes)

    # sort indices on distance to start_fugitive
    labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted, labels_full_sorted_inv = sort_and_filter_nodes(
        graph,
        fugitive_start,
        fugitive_routes,
        police_start,
        t_max)

    # sensor_locations = [(2, 2), (2, 1)]
    # sensor_locations = [(1, 0)]
    sensor_locations = [(2, 0)]
    # sensor_locations = [(2, 1)]
    # sensor_locations = [(0, 1)]
    sensor_locations_labeled = [labels[sensor] for sensor in sensor_locations]

    upper_bounds = []
    for u in range(num_units):
        if len(labels_perunit_sorted[u]) <= 1:
            upper_bounds.append(0.999)
        else:
            upper_bounds.append(len(labels_perunit_sorted[u]) - 0.001)  # different for each unit

    fugitive_routes_labeled = []
    for realization in range(len(fugitive_routes)):
        my_list = []
        for t in range(t_max):
            my_list.append(labels[fugitive_routes[realization][t]])
        fugitive_routes_labeled.append(my_list)
    fugitive_routes_labeled = np.array(fugitive_routes_labeled)
    fugitive_routes_labeled_orig = fugitive_routes_labeled.copy()

    num_intercepted = 0
    police_routes = defaultdict(dict)
    ground_truth_routes = dict()
    interception_dict = dict()

    for rep in range(len(fugitive_routes_labeled_orig)):
    # for rep in range(10):
        print(rep)
        interception_dict[rep] = 0
        fugitive_routes_labeled = fugitive_routes_labeled_orig
        ground_truth_route = fugitive_routes[rep]
        ground_truth_route_labeled = fugitive_routes_labeled[rep]

        ground_truth_routes[rep] = ground_truth_route
        intercepted = False

        # print('ground truth', ground_truth_route)

        for time_step in range(t_max):
            # print('time step: ', time_step)
            if time_step != 0:
                # update police locations (1st step of shortest path to previous target node)
                for u in range(num_units):
                    if police_current[u] != actions[u]:
                        police_current[u] = nx.shortest_path(G=graph,
                                                             source=police_current[u],
                                                             target=actions[u])[1]

                for u, pol_pos in enumerate(police_current):
                    police_routes[rep][u].append(pol_pos)

            else:  # time_step == 0
                police_current = police_start.copy()

                for u, pol_pos in enumerate(police_current):
                    police_routes[rep][u] = [pol_pos]

            # print(time_step, 'police_current: ', police_current)

            # check for interception
            for u, pol in enumerate(police_current):
                # if ground_truth_route[time_step] == pol:
                #     if pol != actions[u]:
                #         print('weird case')

                if ground_truth_route[time_step] == pol == actions[u]:
                    num_intercepted += 1
                    intercepted = True
                    interception_dict[rep] = 1

                    # print(time_step, 'intercepted!')
                    # print(time_step, 'ground truth route: ', ground_truth_route)
                    # print(time_step, 'target nodes: ', actions)
                    # print(time_step, 'police_current', police_current)
                    # print('__________________')

                    break

            if intercepted:
                break

            # check interception of 'ground truth' fugitive route
            sensor_triggered = [sensor == ground_truth_route_labeled[time_step] for sensor in
                                sensor_locations_labeled]

            # if True in sensor_triggered:
                # print(time_step, 'sensor triggered!')

            # filter routes based on sensor info
            fugitive_routes_labeled = recalc_fug_routes(fugitive_routes_labeled, sensor_locations_labeled,
                                                        sensor_triggered, time_step)

            # update reachability {(unit, node): time_to_node, ...}
            # TODO: check this func: maybe time step should be different? can the unit still adjust after the sensor in time?
            tau_uv = unit_ranges(start_units=police_current, U=num_units, G=graph,
                                 labels_full_sorted=labels)

            fugitive_routes_labeled_dicts = []
            for route in fugitive_routes_labeled:
                fugitive_routes_labeled_dicts.append(dict(enumerate(route)))
            # [list(fugitive_routes[i].values()) for i in range(len(fugitive_routes))] (other way around - for DPS and PTO)

            route_data = route_convert(fugitive_routes_labeled_dicts)

            # run optimization
            start_opt = time.time()
            results = optimize(city=graph_type,
                               vary_param='',
                               num_nodes='testgraph3',
                               tau_uv=tau_uv,
                               upper_bounds=upper_bounds,
                               num_units=num_units,
                               route_data=route_data,
                               time_step=time_step,
                               labels=labels,
                               labels_perunit_inv_sorted=labels_perunit_inv_sorted)
            # print(f'optimization took {time.time() - start_opt} seconds')

            # print(results)

            # extract actions for each unit from optimization results
            actions_labeled = [int(np.floor(results[f'pi_{i}'].values[0])) for i in range(num_units)]
            actions = [labels_perunit_inv_sorted[u][i] for u, i in enumerate(actions_labeled)]

            # print(time_step, 'actions: ', actions, 'with expected pct of remaining routes intercepted: ',
            #       (results['num_intercepted'].values[0] / len(fugitive_routes_labeled)))


    print(f'END: pct intercepted: {num_intercepted / len(fugitive_routes)}')
    print(f'END: script took {(time.time() - start_all) / 60} minutes')

    pickle.dump(police_routes,
                open(f"./results/{graph_type}/result_police_routes_{graph_type}_S{len(sensor_locations)}.pkl", "wb"))
    pickle.dump(ground_truth_routes,
                open(f"./results/{graph_type}/result_fug_routes_{graph_type}_S{len(sensor_locations)}.pkl", "wb"))
    pickle.dump(interception_dict,
                open(f"./results/{graph_type}/result_interception_{graph_type}_S{len(sensor_locations)}.pkl", "wb"))

    interception_dict

>>>>>>> d8d05820cb29fb4ce0fcf8122c81cca885774129
