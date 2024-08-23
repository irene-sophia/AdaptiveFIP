import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shutil

from ema_workbench import Model, RealParameter, Constant, Policy, ScalarOutcome, ema_logging
from ema_workbench.em_framework.optimization import GenerationalBorg
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator, SequentialEvaluator
from ema_workbench.em_framework.optimization import ArchiveLogger, OperatorProbabilities

ema_logging.log_to_stderr(ema_logging.INFO)

from DPS.problem_formulation import get_problem_formulation
from network import test_graph
from DPS.sort_and_filter import sort_and_filter_pol_fug as sort_and_filter_nodes
from fugitive_interception_model_S1_U2 import fugitive_interception_model
from DPS.rbf import rbf_cubic

if __name__ == '__main__':
    for seed in range(10):
        try:
            shutil.rmtree('./results/tmp')
        except OSError as e:
            pass

        mode = 'optimization'
        n_realizations = 10
        t_max = 5
        graph_type = 'test_graph_traveltime_3'

        if graph_type == 'test_graph_2':
            graph, labels, labels_inv, pos = test_graph(n_paths=2)
        if graph_type == 'test_graph_traveltime_3':
            graph, labels, labels_inv, pos = test_graph(n_paths=3)

        num_units = 1
        sensor_locations = [(1, 2)]  # , (1, 1)

        # graph = nx.path_graph(7)
        # police_start = 1
        # fugitive_routes = np.ones((n_realizations, t_max)) * 6  # stay at node 6
        # graph = pd.read_pickle(
        #     f"../data/{graph_type}/graph.pkl")
        police_start = pd.read_pickle(
            f"../data/{graph_type}/units_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")
        fugitive_start = pd.read_pickle(
            f"../data/{graph_type}/fugitive_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")
        fugitive_routes = pd.read_pickle(
            f"../data/{graph_type}/fugitive_routes_T{t_max}_R{n_realizations}_U{num_units}.pkl")

        num_units = 2
        police_start = police_start * 2

        labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted, labels_full_sorted_inv = sort_and_filter_nodes(
            graph=graph,
            fugitive_start=fugitive_start,
            fugitive_routes=fugitive_routes,
            police_start=police_start,
            t_max=t_max)

        fugitive_routes_labeled = []
        for realization in range(n_realizations):
            list = []
            for t in range(t_max):
                list.append(labels[fugitive_routes[realization][t]])
            fugitive_routes_labeled.append(list)
        fugitive_routes_labeled = np.array(fugitive_routes_labeled)

        sensor_locations = [labels[loc] for loc in sensor_locations]
        police_start = [labels[police_start[u]] for u in range(len(police_start))]

        model = Model('fugitiveinterception', function=fugitive_interception_model)

        levers, constants, outcomes = get_problem_formulation(mode, graph, police_start, n_realizations, t_max,
                                                              sensor_locations, fugitive_routes_labeled,
                                                              rbf_cubic,
                                                              labels, labels_perunit_sorted,
                                                              labels_perunit_inv_sorted,
                                                              labels_inv)
        model.levers = levers
        model.constants = constants
        model.outcomes = outcomes

        convergence_metrics = [
            ArchiveLogger(
                f"./results/",
                [l.name for l in model.levers],
                [o.name for o in model.outcomes if o.kind != o.INFO],
                base_filename=f"archives_S1_U2_{graph_type}_seed{seed}.tar.gz"
            ),

            # OperatorProbabilities("SBX", 0),
            # OperatorProbabilities("PCX", 1),
            # OperatorProbabilities("DE", 2),
            # OperatorProbabilities("UNDX", 3),
            # OperatorProbabilities("SPX", 4),
            # OperatorProbabilities("UM", 5),
        ]

        with SequentialEvaluator(model) as evaluator:
            results = evaluator.optimize(algorithm=GenerationalBorg,
                                         nfe=5e3, searchover='levers',
                                         epsilons=[0.1, ] * len(model.outcomes),
                                         convergence=convergence_metrics, convergence_freq=10)

        convergence = ArchiveLogger.load_archives(f"./results/archives_S1_U2_{graph_type}_seed{seed}.tar.gz")
        print(results)

        convergence_df = pd.DataFrame()
        for nfe, archive in convergence.items():
            archive['nfe'] = nfe
            convergence_df = pd.concat([convergence_df, archive])
        convergence_df.to_csv(f'./results/archives_S{1}_U{2}_{graph_type}_seed{seed}.csv')
        print(convergence_df)

        # vars = results.iloc[:, :5]  # 2LM + DM
        #
        # mode = 'simulation'
        # model_simulation = Model('fugitiveinterceptionsim', function=fugitive_interception_model)
        #
        # levers, constants, outcomes = get_problem_formulation(mode, graph, police_start, n_realizations, t_max,
        #                                                       sensor_locations, fugitive_routes_labeled,
        #                                                       rbf_cubic,
        #                                                       labels, labels_perunit_sorted,
        #                                                       labels_perunit_inv_sorted,
        #                                                       labels_inv)
        #
        # model_simulation.levers = levers
        # model_simulation.constants = constants
        # model_simulation.outcomes = outcomes
        #
        # policy = [Policy(name=f'DPS_solution_{idx}', c1=row['c1'], r1=row['r1'], w1=row['w1'], w2=row['w2'])
        #           for idx, row in vars.iterrows()]
        #
        # with SequentialEvaluator(model_simulation) as evaluator:
        #     # Run 1000 scenarios for 5 policies
        #     experiments, outcomes = evaluator.perform_experiments(policies=policy)

        # print(outcomes)
