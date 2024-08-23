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
        police_current_opt=None,
        delays=None,
        route_data=None,
        time_step=None,
        **kwargs,
):
    pi_nodes = {}

    for u, value in enumerate(kwargs.values()):
        associated_node = labels_perunit_sorted_inv[u][int(np.floor(value))]
        # reken hier 1 keer de reistijd naar de associated node uit ipv die hele matrix
        travel_time_to_target = nx.shortest_path_length(graph,
                                                      source=police_current_opt[u],
                                                      target=associated_node,
                                                      weight='travel_time',
                                                      method='bellman-ford') + delays[u]
        associated_node = labels[associated_node]
        pi_nodes[u] = (associated_node, travel_time_to_target)

    result = set()
    for u_idx, pi_value in pi_nodes.items():  # for each police unit
        if pi_value[0] not in route_data:
            continue
        for fugitive_time in route_data[pi_value[0]]:
            if fugitive_time[1] >= (pi_value[1] + time_step):  # interception should be after the current time step ofc
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


def optimize(city, graph, delays, num_nodes, police_current_opt, upper_bounds, num_units, num_sensors, route_data, time_step,
             labels, labels_perunit_inv_sorted):
    model = Model("FIPEMA", function=FIP_func)

    model.levers = [RealParameter(f"pi_{u}", 0, upper_bounds[u]) for u in range(num_units)]

    model.constants = model.constants = [
        Constant("route_data", route_data),
        Constant("delays", delays),
        # Constant("tau_uv", tau_uv),
        Constant("labels", labels),
        Constant("labels_perunit_sorted_inv", labels_perunit_inv_sorted),
        Constant("time_step", time_step),
        Constant("graph", graph),
        Constant("police_current_opt", police_current_opt),
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
            nfe=1000,
            searchover="levers",
            convergence=convergence_metrics,
            convergence_freq=100
        )
    print(results)

    del model

    return results


def run_rep(args):
    (rep, graph, graph_type, t_max, fugitive_routes, fugitive_routes_labeled_orig, ground_truth_routes,
     sensor_locations_labeled, num_units, num_sensors, police_start, police_routes, upper_bounds, labels,
     labels_perunit_inv_sorted) = args

    print(rep)

    # t_max is last time stamp of fugitive routes
    t_max = int(((max([max(route.keys()) for i, route in enumerate(fugitive_routes)]) - 1) // 30 + 1) * 30) + 30
    print(t_max)

    # police_planned_path = dict()
    police_planned_path = {u: [node] for u, node in enumerate(police_start)}
    police_progress_on_link = {u: 0 for u in range(num_units)}
    # police_time_left_on_link = {u: 0 for u in range(num_units)}

    fugitive_routes_labeled = fugitive_routes_labeled_orig
    ground_truth_route = fugitive_routes[rep]
    ground_truth_route_labeled = fugitive_routes_labeled[rep]
    print(ground_truth_route)

    ground_truth_routes[rep] = ground_truth_route
    intercepted = False

    for time_step in range((int(t_max / 30) + 1)):
        if time_step == 0:  # time_step == 0
            time_interval = [0, 0]
            police_current = police_start.copy()

            for u, pol_pos in enumerate(police_current):
                police_routes[rep][u] = {0: pol_pos}

        elif time_step != 0:
            time_interval = [(time_step - 1) * 30, time_step * 30]
            # update police locations (1st step of shortest path to previous target node)
            for u in range(num_units):
                t = time_interval[0]
                if police_current[u] != actions[u]:
                    if police_progress_on_link[u] == 0:
                        police_planned_path[u] = nx.shortest_path(G=graph,
                                                                  source=police_current[u],
                                                                  target=actions[u],
                                                                  weight='travel_time',
                                                                  method='bellman-ford')
                        if len(police_planned_path[u]) > 1:
                            del police_planned_path[u][0]  # is source node (= current node)
                    else:
                        link_travel_time_left = min(
                            [float(graph[police_current[u]][police_planned_path[u][0]][i]['travel_time']) for i in
                             graph[police_current[u]][police_planned_path[u][0]]]) - police_progress_on_link[u]

                        if link_travel_time_left > 30:  # time step interval
                            police_progress_on_link[u] += 30
                            t += 31
                        else:
                            t += link_travel_time_left
                            police_progress_on_link[u] = 0
                            police_current[u] = police_planned_path[u][0]
                            police_planned_path[u] = nx.shortest_path(G=graph,
                                                                      source=police_current[u],
                                                                      target=actions[u],
                                                                      weight='travel_time',
                                                                      method='bellman-ford')
                            if len(police_planned_path[u]) > 1:
                                del police_planned_path[u][0]  # is source node (= current node)
                            police_routes[rep][u][t] = police_current[u]
                    try:
                        while ((t <= time_interval[1]) and (police_current[u] != police_planned_path[u][-1])):
                            # print(u, t, 'first statement: ', (t <= time_interval[1]))
                            # print(u, t, 'second statement: ', (police_current[u] != police_planned_path[u][-1]))
                            # print(u, t, 'both: ', ((t <= time_interval[1]) and (police_current[u] != police_planned_path[u][-1])))
                            # t += travel time from police_current to first item from police_planned
                            link_length = min([float(graph[police_current[u]][police_planned_path[u][0]][i]['travel_time']) for i in
                                      graph[police_current[u]][police_planned_path[u][0]]])
                            t += link_length - police_progress_on_link[u]
                            if t > time_interval[1]:
                                police_progress_on_link[u] = link_length - (t - time_interval[1])
                            else:
                                police_current[u] = police_planned_path[u][0]
                                if len(police_planned_path[u]) > 1:
                                    del police_planned_path[u][0]
                                # else:
                                #     pass
                                police_routes[rep][u][t] = police_current[u]
                    except:
                        pass

            # check for interception in the past time interval
            for u, pol in enumerate(police_current):
                if pol == actions[u]:  # if police unit u has arrived at its target node
                    for fug_t, fug_node in {t: node for t, node in ground_truth_route.items() if time_interval[0] <= t <= time_interval[1]}.items():
                        if fug_node == actions[u]:  # if the ground truth route passes the target node in the time interval
                            try:
                                if fug_t >= min([pol_t for pol_t, pol_node in police_routes[rep][u].items() if (pol_node == actions[u])]):  # if police unit u has arrived at its target node before the fug
                                    intercepted = True
                                    print(rep, 'INTERCEPTED')
                                    return rep, intercepted
                            except:
                                pass

        # if intercepted:
        #     return rep, intercepted

        sensor_triggered = [sensor in [node for t, node in ground_truth_route_labeled.items()
                                       if time_interval[0] <= t <= time_interval[1]]
                            for sensor in sensor_locations_labeled]

        # filter routes based on sensor info
        fugitive_routes_labeled = recalc_fug_routes_city(fugitive_routes_labeled, sensor_locations_labeled,
                                                         sensor_triggered, time_interval)
        print('num routes still possible: ', len(fugitive_routes_labeled))
        assert ground_truth_route_labeled in fugitive_routes_labeled

        route_data = route_convert(fugitive_routes_labeled)
        delays = {}
        police_current_opt = []
        for u, prog in police_progress_on_link.items():
            if prog == 0:
                delays[u] = 0
                police_current_opt.append(police_current[u])

            else:
                delays[u] = min([float(graph[police_current[u]][police_planned_path[u][0]][i]['travel_time']) for i in
                         graph[police_current[u]][police_planned_path[u][0]]]) - police_progress_on_link[u]
                police_current_opt.append(police_planned_path[u][0])  # optimize from next node w/ delay

        assert len(police_current) == len(police_current_opt)

        # run optimization
        results = optimize(city=graph_type,
                           graph=graph,
                           delays=delays,
                           num_nodes='city',
                           # tau_uv=tau_uv,
                           police_current_opt=police_current_opt,
                           upper_bounds=upper_bounds,
                           num_units=num_units,
                           num_sensors=num_sensors,
                           route_data=route_data,
                           time_step=time_interval[1],
                           labels=labels,
                           labels_perunit_inv_sorted=labels_perunit_inv_sorted)

        # extract actions for each unit from optimization results
        actions_labeled = [int(np.floor(results[f'pi_{i}'].values[0])) for i in range(num_units)]
        actions = [labels_perunit_inv_sorted[u][i] for u, i in enumerate(actions_labeled)]

        print(actions)

    return rep, intercepted


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

    sensor_locations = pd.read_pickle(
        f"../data/{graph_type}/sp_const_sensors_{city}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
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
    fugitive_routes_labeled_orig = fugitive_routes_labeled.copy()

    num_intercepted = 0
    police_routes = defaultdict(dict)
    ground_truth_routes = dict()
    interception_dict = dict()

    # pool = Pool(processes=cpu_count() - 1)
    with Pool(processes=1) as pool:
    # pool = Pool(processes=3)
        args = []
        # for rep in range(len(fugitive_routes_labeled_orig)):
        for rep in range(3):
            args.append((rep, graph, graph_type, t_max, fugitive_routes, fugitive_routes_labeled_orig, ground_truth_routes,
                            sensor_locations_labeled, num_units, num_sensors, police_start, police_routes, upper_bounds, labels,
                            labels_perunit_inv_sorted))
        time.sleep(1)

        result = pool.map(run_rep, (args))
        interception_dict = {rep: intercepted for (rep, intercepted) in result}

        # pool.close()
        # pool.join()

    num_intercepted = sum(interception_dict.values())
    pct_intercepted = num_intercepted / len(fugitive_routes)

    return pct_intercepted


if __name__ == '__main__':
    n_realizations = 100
    graph_type = 'city'
    city = 'Manhattan'
    num_units = 3
    num_sensors = 3

    pcts = dict()
    for instance in range(1):
        pct_intercepted = run_instance(graph_type, city, n_realizations, num_units, num_sensors, instance)

        print(instance, pct_intercepted)
        pcts[instance] = pct_intercepted

        pickle.dump(pcts, open(
            f'./results/{graph_type}/reopt_sp_const_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl',
            'wb'))

    pickle.dump(pcts, open(
        f'./results/{graph_type}/reopt_sp_const_pctintercepted_R{n_realizations}_U{num_units}_numsensors{num_sensors}.pkl',
        'wb'))

