import sys

# sys.path.append('..')

# import pygraphviz as pgv
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import logging
from random import sample
import random

from ptreeoptperunit.fugitive_interceptions.interception_model_sf_perunit import FugitiveInterception
from ptreeopt.fugitive_interceptions.visualization import plot_result
# from ptreeopt.fugitive_interceptions.networks import rotterdam_graph as graph_func
from ptreeoptperunit.fugitive_interceptions.networks import manhattan_graph as graph_func

from ptreeoptperunit.ptreeopt import PTreeOpt
from ptreeoptperunit.ptreeopt.plotting import *
from ptreeoptperunit.ptreeopt.executors import SequentialExecutor, MPIExecutor, MultiprocessingExecutor


def unit_nodes(start_police, graph, run_length, U):
    unit_nodes = {}
    for u in range(U):
        unit_nodes[u] = list(nx.single_source_shortest_path_length(G=graph, source=start_police[u], cutoff=run_length).keys())

    return unit_nodes


def sort_and_filter_nodes(graph, start_fugitive, route_fugitive, unit_nodes, U):
    """
    Sorts the nodes of a graph on distance to start_fugitive,
    filters the nodes on reachability by the fugitive,
    and returns the updated labels
    """

    distance_dict = {}
    for node in graph.nodes():
        distance_dict[node] = nx.shortest_path_length(G=graph,
                                                      source=start_fugitive,
                                                      target=node,
                                                      weight='travel_time',
                                                      method='dijkstra')

    nx.set_node_attributes(graph, distance_dict, "distance")

    fugitive_nodes = set([y for x in [*route_fugitive] for y in x])

    nodesdict_perunit_sorted = {}
    nodesdict_perunit_inv_sorted = {}
    actions = {}

    for u in range(U):
        police_nodes = set([y for x in [unit_nodes[u]] for y in x])

        node_subset = (list(fugitive_nodes.intersection(police_nodes)))
        subgraph = graph.subgraph(nodes=node_subset)

        nodesdict_perunit_sorted['unit'+str(u)] = {node: 'node' + str(index) for index, node in enumerate(sorted(subgraph.nodes(), key=lambda n: subgraph.nodes[n]['distance']))}
        nodesdict_perunit_inv_sorted['unit'+str(u)] = {'node' + str(index): node for index, node in enumerate(sorted(subgraph.nodes(), key=lambda n: subgraph.nodes[n]['distance']))}

        actions_unit = [f"unit{u}_to_node{i}" for i, node in enumerate(sorted(subgraph.nodes(), key=lambda n: subgraph.nodes[n]['distance']))]
        actions[u] = actions_unit
        # actions.extend(actions_unit)

    return nodesdict_perunit_sorted, nodesdict_perunit_inv_sorted, actions


def run_instance(N, U, R, num_sensors, num_seeds, instance):
    results_instance = pd.DataFrame()

    graph, labels, pos = graph_func(N=N)  # change labels to pointers

    # read in data
    fugitive_start = pd.read_pickle(
        f"../../data/grid/sp_const_fugitive_start_N{N}_T{T}_R{R}_U{U}_numsensors{num_sensors}_instance{instance}.pkl")
    units_start = pd.read_pickle(
        f"../../data/grid/sp_const_units_start_N{N}_T{T}_R{R}_U{U}_numsensors{num_sensors}_instance{instance}.pkl")
    sensor_locations = pd.read_pickle(
        f"../../data/grid/sp_const_sensors_N{N}_T{T}_R{R}_U{U}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes = pd.read_pickle(
        f"../../data/grid/sp_const_fugitive_routes_N{N}_T{T}_R{R}_U{U}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes_db = [list(fugitive_routes[i].values()) for i in
                          range(len(fugitive_routes))]  # to undo timestamps of routes

    unit_nodes_data = unit_nodes(units_start, graph, T, U)
    nodesdict_perunit_sorted, nodesdict_perunit_inv_sorted, actions = sort_and_filter_nodes(graph, fugitive_start,
                                                                                            fugitive_routes_db,
                                                                                            unit_nodes_data, U)

    # print('number of actions, rep ', instance, ': ', len(actions))

    for seed_rep in range(num_seeds):
        # save experiment parameters
        random.seed(seed_rep)
        np.random.seed(seed_rep)

        model = FugitiveInterception(T, U, R, graph=graph,
                                     units_start=units_start, nodesdict_perunit_sorted=nodesdict_perunit_inv_sorted,
                                     fugitive_start=fugitive_start, fugitive_routes_db=fugitive_routes_db,
                                     num_sensors=num_sensors, sensor_locations=sensor_locations,
                                     seed=seed_rep)

        algorithm = PTreeOpt(model.f, num_units=U,
                             feature_bounds=[[0, T]] + [[0.5, 1.5]] * num_sensors,  # indicators
                             feature_names=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],  # indicator names
                             discrete_features=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],
                             discrete_actions=True,
                             action_names=actions,
                             mu=20,  # 20
                             cx_prob=0.70,
                             population_size=100,
                             max_depth=20  # 5
                             )

        logging.basicConfig(level=logging.INFO,
                            format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

        best_solution, best_score, snapshots = algorithm.run(max_nfe=100000,
                                                             log_frequency=100,
                                                             snapshot_frequency=100,
                                                             executor=MultiprocessingExecutor())

        # pickle.dump(snapshots, open('results/sorted_filtered/snapshots_rep{}_seed{}.pkl'.format(instance, seed_rep), 'wb'))
        #
        # # P = snapshots['best_P'][-1]  #best policy tree
        # colors = {f"unit{u}_to_node{i}": 'lightgrey' for u in range(U) for i, _ in enumerate(graph.nodes)}
        # graphviz_export(best_solution, 'figs/sorted_filtered/optimaltree_rep{}_seed{}.png'.format(instance, seed_rep), colordict=colors)  # creates one SVG
        #
        # # model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
        # #                          num_sensors=num_sensors, sensor_locations=sensor_locations)

        convergence_df = pd.DataFrame(snapshots)
        convergence_df['instance'] = instance
        convergence_df['seed'] = seed_rep
        convergence_df['manhattan_diam'] = N
        convergence_df['num_units'] = U
        convergence_df['num_sensors'] = num_sensors

        convergence_df.to_csv(f'results/100k_nfe/results_N{N}_T{T}_R{R}_U{U}_numsensors{num_sensors}_instance{instance}_seed{seed_rep}.csv',
                                index=False)

        results_instance = pd.concat([results_instance, convergence_df])

        results_instance.to_csv(f'results/100k_nfe/results_N{N}_T{T}_R{R}_U{U}_numsensors{num_sensors}_instance{instance}_uptoseed{seed_rep}.csv',
                                index=False)

        # print('finished seed ', seed_rep + 1, 'of repetition ', instance + 1, '(experiment number ',
        #       (instance * num_seeds) + seed_rep + 1, 'of ', (num_seeds * num_repetitions), ').')

        # results_df, success = model.f(best_solution, mode='simulation')  # re-initializes! only use for visuals
        # print('Simulation: interception percentage: ', (sum(success.values()) * 100)/R)
        # print(results_df['policy'])

        # plot_result(graph, pos, T, U, R, results_df, success, sensor_locations, labels)

    results_instance.to_csv(f'results/100k_nfe/results_N{N}_T{T}_R{R}_U{U}_numsensors{num_sensors}_instance{instance}.csv',
                            index=False)

    return results_instance


if __name__ == '__main__':
    N = 10
    T = int(5 + (0.5 * N))
    U = 3
    R = 100
    num_sensors = 10

    num_repetitions = 1
    num_seeds = 10

    # for instance in range(num_repetitions):
    for instance in [1]:
        results_instance = run_instance(N, U, R, num_sensors, num_seeds, instance)

        print(results_instance)
