import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
from collections import defaultdict

from ema_workbench import Model, RealParameter, Constant, Policy, ScalarOutcome, ema_logging
from ema_workbench.em_framework.optimization import GenerationalBorg
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator, SequentialEvaluator
from ema_workbench.em_framework.optimization import ArchiveLogger, OperatorProbabilities

from DPS.problem_formulation import get_problem_formulation_rbfs
from network import test_graph_traveltime as test_graph
from DPS.sort_and_filter import sort_and_filter_pol_fug_city as sort_and_filter_nodes
from fugitive_interception_model_grid import fugitive_interception_model
from DPS.rbf import rbf_gaussian, rbf_cubic, rbf_linear

ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == '__main__':
    graph_type = 'test_graph_3'
    manhattan_diameter = 10
    n_realizations = 9
    num_units = 1
    num_sensors = 1
    t_max = 6
    mode = 'optimization'
    num_rbfs = 6
    rbf = rbf_linear
    seed = 0

    nfe_of_convergence = []
    for instance in range(1):
        try:
            shutil.rmtree('./results/tmp')
        except OSError as e:
            pass

        # if graph_type == 'manhattan':
        #     graph, labels, labels_inv, pos = manhattan_graph(manhattan_diameter)
        if graph_type == 'test_graph_2':
            graph, labels, labels_inv, pos = test_graph(n_paths=2, travel_time=1)
        if graph_type == 'test_graph_3':
            graph, labels, labels_inv, pos = test_graph(n_paths=3, travel_time=1)

        # graph = nx.path_graph(7)
        # police_start = 1
        # fugitive_routes = np.ones((n_realizations, t_max)) * 6  # stay at node 6
        # graph = pd.read_pickle(
        #     f"../data/{graph_type}/graph.pkl")
        # police_start = pd.read_pickle(
        #     f"../../data/{graph_type}/units_start_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
        # fugitive_start = pd.read_pickle(
        #     f"../../data/{graph_type}/fugitive_start_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
        # fugitive_routes_nodict = pd.read_pickle(
        #     f"../../data/{graph_type}/fugitive_routes_T{t_max}_R{n_realizations}_U{num_units}.pkl")
        # sensor_locations = pd.read_pickle(
        #     f"../../data/{graph_type}/sensors_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")

        police_start = [(5, 1)]
        fugitive_start = (0, 1)
        sensor_locations = [(2, 2)]
        sensor_locations = [(0, 1)]
        fugitive_routes_nodict = ([[(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)]] * 3 +
                                  [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]] * 3 +
                                  [[(0, 1), (1, 2), (2, 2), (3, 2), (4, 2)]] * 3)
        fugitive_routes = []
        for route in fugitive_routes_nodict:
            fugitive_routes.append(dict(enumerate(route)))

        labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted, labels_full_sorted_inv = sort_and_filter_nodes(
            graph=graph,
            fugitive_start=fugitive_start,
            fugitive_routes=fugitive_routes,
            police_start=police_start,
            t_max=t_max)

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

        for seed in range(100):
            convergence_metrics = [
                ArchiveLogger(
                    f"./results/",
                    [l.name for l in model.levers],
                    [o.name for o in model.outcomes if o.kind != o.INFO],
                    base_filename=f"archives_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_{rbf.__name__}_numrbf_{num_rbfs}_{graph_type}_instance{instance}_seed{seed}.tar.gz"
                ),
                # OperatorProbabilities("SBX", 0),
                # OperatorProbabilities("PCX", 1),
                # OperatorProbabilities("DE", 2),
                # OperatorProbabilities("UNDX", 3),
                # OperatorProbabilities("SPX", 4),
                # OperatorProbabilities("UM", 5),
            ]
            with SequentialEvaluator(model) as evaluator:
            # with MultiprocessingEvaluator(model, n_processes=12) as evaluator:
                results = evaluator.optimize(algorithm=GenerationalBorg,
                                             nfe=2e3, searchover='levers',
                                             epsilons=[1/n_realizations, ] * len(model.outcomes),
                                             convergence=convergence_metrics, convergence_freq=10)

            convergence = ArchiveLogger.load_archives(f"./results/archives_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_{rbf.__name__}_numrbf_{num_rbfs}_{graph_type}_instance{instance}_seed{seed}.tar.gz")
            print(results)

            convergence_df = pd.DataFrame()
            for nfe, archive in convergence.items():
                archive['nfe'] = nfe
                convergence_df = pd.concat([convergence_df, archive])
            del convergence_df['Unnamed: 0']
            convergence_df.to_csv(f'./results/convergence_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_{rbf.__name__}_numrbf_{num_rbfs}_{graph_type}_instance{instance}_seed{seed}.csv')
            convergence_df = convergence_df.sort_values(by=['nfe'])
            print(convergence_df)

            nfe_of_convergence.append(convergence_df[convergence_df.not_intercepted <= 0.8].iloc[0]['nfe'])

        # vars = results.iloc[:, :3]  # 2LM + DM
        #
        # mode = 'simulation'
        # model_simulation = Model('fugitiveinterceptionsim', function=fugitive_interception_model)
        #
        # levers, constants, outcomes = get_problem_formulation_rbfs(mode, graph, police_start, n_realizations, t_max,
        #                                                       sensor_locations,
        #                                                       fugitive_routes_labeled,
        #                                                       rbf_linear, num_rbfs,
        #                                                       labels, labels_perunit_sorted,
        #                                                       labels_perunit_inv_sorted,
        #                                                       labels_inv,
        #                                                       sensor_detections,)
        #
        # model_simulation.levers = levers
        # model_simulation.constants = constants
        # model_simulation.outcomes = outcomes

        # policy = [Policy(name=f'DPS_solution_{idx}', c1=row['c1'], r1=row['r1'], w1=row['w1'])
        #           for idx, row in vars.iterrows()]

        # policy = [Policy(name=f'DPS_solution_{idx}', c1=row['c1'], r1=row['r1'], c2=row['c2'], r2=row['r2'], w1=row['w1'], w2=row['w2'], w3=row['w3'], w4=row['w4'])
        #           for idx, row in vars.iterrows()]

        # policy = [Policy(name=f'DPS_solution_{idx}', c1=row['c1'],  c2=row['c2'], c3=row['c3'], c4=row['c4'], c5=row['c5'], c6=row['c6'],
        #                  r1=row['r1'], r2=row['r2'], r3=row['r3'], r4=row['r4'], r5=row['r5'], r6=row['r6'],
        #                  w1=row['w1'], w2=row['w2'], w3=row['w3'], w4=row['w4'], w5=row['w5'], w6=row['w6'])
        #           for idx, row in vars.iterrows()]

        # with SequentialEvaluator(model_simulation) as evaluator:
        #     experiments, outcomes = evaluator.perform_experiments(policies=policy)

        # print(outcomes)

    print('avg nfe to solution: ', np.mean(nfe_of_convergence))
