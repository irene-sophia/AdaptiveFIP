import sys

sys.path.append('..')

# import pygraphviz as pgv
import numpy as np
import networkx as nx
import pickle
import logging
from random import sample
import random

from fugitive_interceptions.interception_model_sorted import FugitiveInterception
from fugitive_interceptions.visualization import plot_result
# from fugitive_interceptions.networks import rotterdam_graph as graph_func
from fugitive_interceptions.networks import manhattan_graph as graph_func

from ptreeopt import PTreeOpt
from ptreeopt.plotting import *


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


N = 10
T = 10
U = 3
R = 20
num_sensors = 3




if __name__ == '__main__':
    num_repetitions = 10
    num_seeds = 10

    for rep in range(num_repetitions):
        graph, labels, pos = graph_func(N=N)  # change labels to pointers
        # open pickles from prev experiment to guarantee same starting configuration
        fugitive_start = pd.read_pickle(r'results/start_fug_rep{}_seed0.pkl'.format(rep))
        units_start = pd.read_pickle(r'results/start_units_rep{}_seed0.pkl'.format(rep))
        sensor_locations = pd.read_pickle(r'results/sensors_rep{}_seed0.pkl'.format(rep))
        fugitive_routes_db = pd.read_pickle(r'results/routes_fug_rep{}_seed0.pkl'.format(rep))

        for seed_rep in range(num_seeds):
            # save experiment parameters
            random.seed(seed_rep)

            pickle.dump(sensor_locations, open('results/sorted/sensors_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))
            pickle.dump(units_start, open('results/sorted/start_units_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))
            pickle.dump(fugitive_start, open('results/sorted/start_fug_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))
            pickle.dump(fugitive_routes_db, open('results/sorted/routes_fug_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))

            model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
                                         fugitive_routes_db=fugitive_routes_db,
                                         num_sensors=num_sensors, sensor_locations=sensor_locations, seed=seed_rep)

            algorithm = PTreeOpt(model.f,
                                 feature_bounds=[[0, T]] + [[0.5, 1.5]] * num_sensors,  # indicators
                                 feature_names=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],  # indicator names
                                 discrete_features=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],
                                 discrete_actions=True,
                                 action_names=[f"unit{u}_to_node{i}" for u in range(U) for i, _ in enumerate(graph.nodes)],
                                 mu=20,  # 20
                                 cx_prob=0.70,
                                 population_size=100,
                                 max_depth=10  # 5
                                 )

            logging.basicConfig(level=logging.INFO,
                                format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

            best_solution, best_score, snapshots = algorithm.run(max_nfe=100000,
                                                                 log_frequency=100,
                                                                 snapshot_frequency=100)

            pickle.dump(snapshots, open('results/sorted/snapshots_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))

            # P = snapshots['best_P'][-1]  #best policy tree
            colors = {f"unit{u}_to_node{i}": 'lightgrey' for u in range(U) for i, _ in enumerate(graph.nodes)}
            graphviz_export(best_solution, 'figs/sorted/optimaltree_rep{}_seed{}.png'.format(rep, seed_rep), colordict=colors)  # creates one SVG

            # model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
            #                          num_sensors=num_sensors, sensor_locations=sensor_locations)

            print('finished seed ', seed_rep+1, 'of repetition ', rep+1, '(experiment number ',
                  (rep*num_seeds)+seed_rep+1, 'of ', (num_seeds*num_repetitions), ').')
            # results_df, success = model.f(best_solution, mode='simulation')  # re-initializes! only use for visuals
            # print('Simulation: interception percentage: ', (sum(success.values()) * 100)/R)
            # print(results_df['policy'])

            # plot_result(graph, pos, T, U, R, results_df, success, sensor_locations, labels)

            # TODO: random.seed