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
from network import manhattan_graph
from DPS.sort_and_filter import sort_and_filter_pol_fug_city as sort_and_filter_nodes
from fugitive_interception_model_city import fugitive_interception_model
from DPS.rbf import rbf_gaussian, rbf_cubic, rbf_linear
import sys

ema_logging.log_to_stderr(ema_logging.INFO)


def run_dps(nfe=5e4, graph_type='manhattan', manhattan_diameter=10, n_realizations=10, num_units=1, num_sensors=1,
            num_rbfs=1, rbf=None, instance=0, seed=0):
    t_max = int(5 + (0.5 * manhattan_diameter))
    mode = 'optimization'

    try:
        shutil.rmtree('./results/tmp')
    except OSError as e:
        pass

    filepath = f"../../data/city/graphs/{city}.graph.graphml"
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
        f"../../data/{graph_type}/sp_const_units_start_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_start = pd.read_pickle(
        f"../../data/{graph_type}/sp_const_fugitive_start_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    fugitive_routes = pd.read_pickle(
        f"../../data/{graph_type}/sp_const_fugitive_routes_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")
    sensor_locations = pd.read_pickle(
        f"../../data/{graph_type}/sp_const_sensors_N{manhattan_diameter}_T{t_max}_R{n_realizations}_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl")

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

    sensor_detections = {i: np.zeros((n_realizations, t_max/30)) for i, _ in enumerate(sensor_locations)}

    # sensors stay flipped
    for timestep in range(int(t_max / 30)):
        t_interval = timestep * 30
        nodes_interval = [[v for k, v in fugitive_routes_labeled[realization].items() if k <= t_interval] for
                          realization in
                          range(n_realizations)]

        for i, sensor_loc in enumerate(sensor_locations):
            for realization in range(n_realizations):
                sensor_detections[i][realization, timestep] = sensor_loc in nodes_interval[realization]

    model = Model('fugitiveinterception', function=fugitive_interception_model)

    levers, constants, outcomes = get_problem_formulation_rbfs(mode, graph, police_start, n_realizations, t_max,
                                                               sensor_locations,
                                                               fugitive_routes_labeled,
                                                               rbf, num_rbfs,
                                                               labels, labels_perunit_sorted,
                                                               labels_perunit_inv_sorted,
                                                               labels_inv,
                                                               sensor_detections, )
    model.levers = levers
    model.constants = constants
    model.outcomes = outcomes

    convergence_metrics = [
        ArchiveLogger(
            f"./results/",
            [l.name for l in model.levers],
            [o.name for o in model.outcomes if o.kind != o.INFO],
            base_filename=f"archives_{city}_T{t_max}_R{n_realizations}_U{num_units}_S{num_sensors}_{rbf.__name__}_numrbf{num_rbfs}_instance{instance}_seed{seed}.tar.gz"
        ),

        # OperatorProbabilities("SBX", 0),
        # OperatorProbabilities("PCX", 1),
        # OperatorProbabilities("DE", 2),
        # OperatorProbabilities("UNDX", 3),
        # OperatorProbabilities("SPX", 4),
        # OperatorProbabilities("UM", 5),
    ]
    # with SequentialEvaluator(model) as evaluator:
    with MultiprocessingEvaluator(model, n_processes=12) as evaluator:
        results = evaluator.optimize(algorithm=GenerationalBorg,
                                     nfe=nfe, searchover='levers',
                                     epsilons=[1 / n_realizations, ] * len(model.outcomes),
                                     convergence=convergence_metrics, convergence_freq=10)

    convergence = ArchiveLogger.load_archives(
        f"./results/sp_const_archives_{city}_T{t_max}_R{n_realizations}_U{num_units}_S{num_sensors}_{rbf.__name__}_numrbf{num_rbfs}_instance{instance}_seed{seed}.tar.gz")
    print(results)

    convergence_df = pd.DataFrame()
    for nfe, archive in convergence.items():
        archive['nfe'] = nfe
        convergence_df = pd.concat([convergence_df, archive])
    convergence_df.to_csv(
        f'./results/sp_const_convergence_{city}_T{t_max}_R{n_realizations}_U{num_units}_S{num_sensors}_{rbf.__name__}_numrbf{num_rbfs}_instance{instance}_seed{seed}.csv')


if __name__ == '__main__':
    num_rbfs = 2
    rbf = rbf_linear
    nfe = 1e5

    num_units = int(sys.argv[-4])
    num_sensors = int(sys.argv[-3])
    n_realizations = int(sys.argv[-2])
    manhattan_diameter = int(sys.argv[-1])

    for city in ['Manhattan', 'Winterwsijk', 'Utrecht']:
        for num_sensors in [3, 10]:
            for num_units in [3, 10]:
                for instance in range(10):
                    for seed in range(10):

                        run_dps(nfe=nfe,
                                graph_type='city',
                                n_realizations=n_realizations,
                                num_units=num_units,
                                num_sensors=num_sensors,
                                num_rbfs=num_rbfs,
                                rbf=rbf,
                                instance=instance,
                                seed=seed
                                )

