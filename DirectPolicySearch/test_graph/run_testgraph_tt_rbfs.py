import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
from collections import defaultdict

from ema_workbench import Model, RealParameter, Constant, Policy, ScalarOutcome, ema_logging
from ema_workbench.em_framework.optimization import GenerationalBorg, EpsNSGAII
from ema_workbench.em_framework.optimization import SingleObjectiveBorgWithArchive
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator, SequentialEvaluator
from ema_workbench.em_framework.optimization import ArchiveLogger, OperatorProbabilities

from DPS.problem_formulation import get_problem_formulation_rbfs
from network import test_graph_traveltime as test_graph
from DPS.sort_and_filter import sort_and_filter_pol_fug_city as sort_and_filter_nodes
from fugitive_interception_model_tt_rbfs import fugitive_interception_model
from DPS.rbf import rbf_gaussian, rbf_cubic, rbf_linear

ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == '__main__':
    for num_sensors in [1]:
        for rbf in [rbf_linear]:
            for num_rbfs in [2]:
                for seed in range(1):
                    mode = 'optimization'
                    n_realizations = 10
                    t_max = 6
                    graph_type = 'test_graph_traveltime_3'

                    try:
                        shutil.rmtree(f'./results/{graph_type}/tmp')
                    except OSError as e:
                        pass

                    if graph_type == 'test_graph_traveltime_2':
                        graph, labels, labels_inv, pos = test_graph(n_paths=2, travel_time=1)
                    if graph_type == 'test_graph_traveltime_3':
                        graph, labels, labels_inv, pos = test_graph(n_paths=3, travel_time=1)

                    num_units = 1
                    if num_sensors == 1:
                        # sensor_locations = [(2, 2)]
                        sensor_locations = [(2, 0)]
                    if num_sensors == 2:
                        sensor_locations = [(1, 2), (2, 2)]

                    # graph = nx.path_graph(7)
                    # police_start = 1
                    # fugitive_routes = np.ones((n_realizations, t_max)) * 6  # stay at node 6
                    # graph = pd.read_pickle(
                    #     f"../data/{graph_type}/graph.pkl")
                    police_start = pd.read_pickle(
                        f"../../data/{graph_type}/units_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")

                    fugitive_start = pd.read_pickle(
                        f"../../data/{graph_type}/fugitive_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")
                    # fugitive_routes = pd.read_pickle(
                    #     f"../../data/{graph_type}/fugitive_routes_T{t_max}_R{n_realizations}_U{num_units}.pkl")

                    fugitive_routes_nodict = ([[(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)]] * 33 +
                                              [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]] * 33 +
                                              [[(0, 1), (1, 2), (2, 2), (3, 2), (4, 2)]] * 33)

                    fugitive_routes = []
                    for route in fugitive_routes_nodict:
                        fugitive_routes.append(dict(enumerate(route)))

                    num_units = 1
                    police_start = police_start

                    labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted, labels_full_sorted_inv = sort_and_filter_nodes(
                        graph=graph,
                        fugitive_start=fugitive_start,
                        fugitive_routes=fugitive_routes,
                        police_start=police_start,
                        t_max=t_max)

                    n_realizations = len(fugitive_routes)
                    fugitive_routes_labeled = defaultdict(dict)
                    for realization in range(n_realizations):
                        for t in fugitive_routes[realization].keys():
                            fugitive_routes_labeled[realization][t] = labels[fugitive_routes[realization][t]]

                    sensor_locations = [labels[loc] for loc in sensor_locations]
                    police_start = [labels[police_start[u]] for u in range(len(police_start))]

                    sensor_detections = {i: np.zeros((n_realizations, t_max)) for i, _ in enumerate(sensor_locations)}

                    # sensors stay flipped
                    for t_interval in range(t_max):
                        nodes_interval = [[v for k, v in fugitive_routes_labeled[realization].items() if k <= t_interval] for realization in
                                          range(n_realizations)]

                        for i, sensor_loc in enumerate(sensor_locations):
                            for realization in range(n_realizations):
                                sensor_detections[i][realization, t_interval] = sensor_loc in nodes_interval[realization]

                    model = Model('fugitiveinterception', function=fugitive_interception_model)

                    levers, constants, outcomes = get_problem_formulation_rbfs(mode, graph, police_start, n_realizations, t_max,
                                                                          sensor_locations,
                                                                          fugitive_routes_labeled,
                                                                          rbf, num_rbfs,
                                                                          labels, labels_perunit_sorted,
                                                                          labels_perunit_inv_sorted,
                                                                          labels_inv,
                                                                          sensor_detections,)
                    model.levers = levers
                    model.constants = constants
                    model.outcomes = outcomes

                    convergence_metrics = [
                        ArchiveLogger(
                            f"./results/{graph_type}/",
                            [l.name for l in model.levers],
                            [o.name for o in model.outcomes if o.kind != o.INFO],
                            base_filename=f"archives_tt_U{len(police_start)}_S{len(sensor_locations)}_{rbf.__name__}_numrbf_{num_rbfs}_{graph_type}_seed{seed}.tar.gz"
                        ),

                        # OperatorProbabilities("SBX", 0),
                        # OperatorProbabilities("PCX", 1),
                        # OperatorProbabilities("DE", 2),
                        # OperatorProbabilities("UNDX", 3),
                        # OperatorProbabilities("SPX", 4),
                        # OperatorProbabilities("UM", 5),
                    ]

                    with MultiprocessingEvaluator(model, n_processes=8) as evaluator:
                        results = evaluator.optimize(algorithm=SingleObjectiveBorgWithArchive,
                                                     nfe=5e3, searchover='levers',
                                                     epsilons=[0.1, ] * len(model.outcomes),
                                                     convergence=convergence_metrics, convergence_freq=10)

                    convergence = ArchiveLogger.load_archives(f"./results/{graph_type}/archives_tt_U{len(police_start)}_S{len(sensor_locations)}_{rbf.__name__}_numrbf_{num_rbfs}_{graph_type}_seed{seed}.tar.gz")
                    print(results)

                    convergence_df = pd.DataFrame()
                    for nfe, archive in convergence.items():
                        archive['nfe'] = nfe
                        convergence_df = pd.concat([convergence_df, archive])
                    convergence_df.to_csv(f'./results/{graph_type}/archives_tt_U{len(police_start)}_S{len(sensor_locations)}_{rbf.__name__}_numrbf{num_rbfs}_{graph_type}_seed{seed}.csv')
                    print(convergence_df)

                    lm = len(sensor_locations) * num_rbfs
                    dm = len(police_start) * num_rbfs
                    num_vars = 2*lm + dm
                    vars = results.iloc[:, :num_vars]  # 2LM + DM

                    mode = 'simulation'
                    model_simulation = Model('fugitiveinterceptionsim', function=fugitive_interception_model)

                    levers, constants, outcomes = get_problem_formulation_rbfs(mode, graph, police_start, n_realizations, t_max,
                                                                          sensor_locations,
                                                                          fugitive_routes_labeled,
                                                                          rbf_linear, num_rbfs,
                                                                          labels, labels_perunit_sorted,
                                                                          labels_perunit_inv_sorted,
                                                                          labels_inv,
                                                                          sensor_detections,)

                    model_simulation.levers = levers
                    model_simulation.constants = constants
                    model_simulation.outcomes = outcomes

                    policy = [Policy(name=f'DPS_solution_{idx}', ) for idx, row in vars.iterrows()]

                    policy = [Policy(name=f'DPS_solution_{idx}', c1=row['c1'], r1=row['r1'], w1=row['w1'])
                              for idx, row in vars.iterrows()]

                    # policy = [Policy(name=f'DPS_solution_{idx}', c1=row['c1'], r1=row['r1'], c2=row['c2'], r2=row['r2'], w1=row['w1'], w2=row['w2'], w3=row['w3'], w4=row['w4'])
                    #           for idx, row in vars.iterrows()]

                    # policy = [Policy(name=f'DPS_solution_{idx}', c1=row['c1'],  c2=row['c2'], c3=row['c3'], c4=row['c4'], c5=row['c5'], c6=row['c6'],
                    #                  r1=row['r1'], r2=row['r2'], r3=row['r3'], r4=row['r4'], r5=row['r5'], r6=row['r6'],
                    #                  w1=row['w1'], w2=row['w2'], w3=row['w3'], w4=row['w4'], w5=row['w5'], w6=row['w6'])
                    #           for idx, row in vars.iterrows()]

                    with SequentialEvaluator(model_simulation) as evaluator:
                        experiments, outcomes = evaluator.perform_experiments(policies=policy)

