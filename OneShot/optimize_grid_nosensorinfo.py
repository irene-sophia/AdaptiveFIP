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


def FIP_func(
        tau_uv=None,
        labels=None,
        labels_perunit_sorted_inv=None,
        route_data=None,
        t_max=None,
        # time_step=None,
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
            if fugitive_time[1] > (pi_value[1]):  #  <= (t_max)

                result.add(fugitive_time[0])  # add route index to intercepted routes
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


def optimize(city, num_nodes, tau_uv, upper_bounds, num_units, route_data, t_max,
             labels, labels_perunit_inv_sorted):
    model = Model("FIPEMA", function=FIP_func)

    model.levers = [RealParameter(f"pi_{u}", 0, upper_bounds[u]) for u in range(num_units)]

    model.constants = model.constants = [
        Constant("route_data", route_data),
        Constant("t_max", t_max),
        Constant("tau_uv", tau_uv),
        Constant("labels", labels),
        Constant("labels_perunit_sorted_inv", labels_perunit_inv_sorted),
        # Constant("time_step", time_step),
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


def run_instance(graph_type, n_realizations, manhattan_diam, num_units, num_sensors, instance):
    t_max = int(5 + (0.5 * manhattan_diam))

    graph, labels, labels_inv, pos = graph_func(manhattan_diam)

    police_start = pd.read_pickle(
        f"../data/{graph_type}/const_units_start_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_start = pd.read_pickle(
        f"../data/{graph_type}/const_fugitive_start_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes = pd.read_pickle(
        f"../data/{graph_type}/const_fugitive_routes_N{manhattan_diam}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes = [list(fugitive_routes[i].values()) for i in range(len(fugitive_routes))]  # to undo timestamps of routes

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

    # update reachability {(unit, node): time_to_node, ...}
    tau_uv = unit_ranges(start_units=police_start, U=num_units, G=graph,
                         labels_full_sorted=labels)

    fugitive_routes_labeled_dicts = []
    for route in fugitive_routes_labeled:
        fugitive_routes_labeled_dicts.append(dict(enumerate(route)))
    # [list(fugitive_routes[i].values()) for i in range(len(fugitive_routes))] (other way around - for DPS and PTO)

    route_data = route_convert(fugitive_routes_labeled_dicts)

    # run optimization
    start_opt = time.time()
    results = optimize(city=graph_type,
                       num_nodes='testgraph3',
                       tau_uv=tau_uv,
                       upper_bounds=upper_bounds,
                       num_units=num_units,
                       route_data=route_data,
                       t_max=t_max,
                       # time_step=time_step,
                       labels=labels,
                       labels_perunit_inv_sorted=labels_perunit_inv_sorted)
    # print(f'optimization took {time.time() - start_opt} seconds')
    result_num_intercepted = results['num_intercepted'][0]
    print(result_num_intercepted)
    # print(results)

    pct_intercepted = result_num_intercepted / len(fugitive_routes)

    return pct_intercepted


if __name__ == '__main__':
    random.seed(112)
    np.random.seed(112)

    n_realizations = 100
    graph_type = 'grid'
    manhattan_diam = 10
    t_max = int(5 + (0.5 * manhattan_diam))

    num_units = 3
    num_sensors = 3

    pcts = dict()
    for instance in range(10):
        pct_intercepted = run_instance(graph_type, n_realizations, manhattan_diam, num_units, num_sensors, instance)

        pickle.dump(pct_intercepted, open(
            f'./results/grid/const_nosensorinfo_pct_intercepted_N{manhattan_diam}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl',
            'wb'))

        print(instance, pct_intercepted)
        pcts[instance] = pct_intercepted

    pickle.dump(pcts, open(
        f'./results/grid/const_nosensorinfo_pct_intercepted_N{manhattan_diam}_R{n_realizations}_U{num_units}_numsensors{num_sensors}.pkl', 'wb'))
