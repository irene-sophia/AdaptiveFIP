import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from ema_workbench import Model, RealParameter, Constant, ScalarOutcome, ema_logging
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator, SequentialEvaluator

ema_logging.log_to_stderr(ema_logging.INFO)

from sort_and_filter import sort_and_filter_pol_fug as sort_and_filter_nodes
from fugitive_interception_model import fugitive_interception_model, rbf_test, rbf_cubic

if __name__ == '__main__':
    n_realizations = 3
    t_max = 5

    graph = nx.path_graph(7)
    police_start = 1
    fugitive_routes = np.ones((n_realizations, t_max)) * 6  # stay at node 6
    sensor_location = 6  # sensor is at fugitive start?

    labels_perunit_sorted, labels_perunit_inv_sorted, _ = sort_and_filter_nodes(graph=graph,
                                                                                fugitive_start=fugitive_routes[0][0],
                                                                                fugitive_routes=fugitive_routes,
                                                                                police_start=police_start,
                                                                                t_max=t_max)

    model = Model('fugitiveinterception', function=fugitive_interception_model)

    model.levers = [
        RealParameter("c1", -1.0, 1.0),
        RealParameter("r1", 0.0, 1.0),
        RealParameter("w1", 0.0, 1.0),
    ]

    model.constants = [
        Constant("graph", graph),
        Constant("police_start", police_start),
        Constant("n_realizations", n_realizations),
        Constant("t_max", t_max),
        Constant("sensor_location", sensor_location),
        Constant("fugitive_routes", fugitive_routes),
        Constant("rbf", rbf_cubic),
        Constant("labels_perunit_sorted", labels_perunit_sorted),
        Constant("labels_perunit_inv_sorted", labels_perunit_inv_sorted)
    ]

    model.outcomes = [
        ScalarOutcome('not_intercepted', kind=ScalarOutcome.MINIMIZE)
    ]

    with SequentialEvaluator(model) as evaluator:
        results = evaluator.optimize(nfe=5e3, searchover='levers',
                                     epsilons=[0.1, ] * len(model.outcomes))

    print(results)
