from ema_workbench import RealParameter, IntegerParameter, Constant, ScalarOutcome, ArrayOutcome


def get_problem_formulation(mode, graph, police_start, n_realizations, t_max, sensor_location, fugitive_routes_labeled,
                            rbf_cubic, num_rbfs,
                            labels_full_sorted, labels_perunit_sorted, labels_perunit_inv_sorted,
                            labels_full_sorted_inv, sensor_detections=None):

    if isinstance(sensor_location, int):
        levers = [RealParameter(f"c1", -1.0, 1.0)] + \
             [RealParameter(f"r1", 0.0001, 1.0)] + \
             [RealParameter(f"w{i + 1}", 0.0, 1.0) for i, _ in enumerate(police_start)]
    else:
        levers = [RealParameter(f"c{i + 1}", -1.0, 1.0) for i, _ in enumerate(sensor_location)] + \
                 [RealParameter(f"r{i + 1}", 0.0001, 1.0) for i, _ in enumerate(sensor_location)] + \
                 [RealParameter(f"w{i + 1}", 0.0, 1.0) for i, _ in enumerate(police_start)]

    constants = [
        Constant("mode", mode),
        Constant("graph", graph),
        Constant("police_start", police_start),
        Constant("n_realizations", n_realizations),
        Constant("t_max", t_max),
        Constant("sensor_location", sensor_location),
        Constant("sensor_detections", sensor_detections),
        Constant("fugitive_routes", fugitive_routes_labeled),
        Constant("rbf", rbf_cubic),
        Constant("labels_perunit_sorted", labels_perunit_sorted),
        Constant("labels_perunit_inv_sorted", labels_perunit_inv_sorted),
        Constant("labels_full_sorted", labels_full_sorted),
        Constant("labels_full_sorted_inv", labels_full_sorted_inv)
    ]

    if mode == 'optimization':
        outcomes = [
            ScalarOutcome('not_intercepted', kind=ScalarOutcome.MINIMIZE)
        ]

    elif mode == 'simulation':
        outcomes = [
            ArrayOutcome('at'),
            ArrayOutcome('police_paths')
        ]

    return levers, constants, outcomes


def get_problem_formulation_rbfs(mode, graph, police_start, n_realizations, t_max, sensor_location, fugitive_routes_labeled,
                            rbf_cubic, num_rbfs,
                            labels_full_sorted, labels_perunit_sorted, labels_perunit_inv_sorted,
                            labels_full_sorted_inv, sensor_detections=None):
    lm = len(sensor_location) * num_rbfs
    dm = len(police_start) * num_rbfs

    if isinstance(sensor_location, int):
        levers = [RealParameter(f"c1", -1.0, 1.0)] + \
                 [RealParameter(f"r1", 0.0001, 1.0)] + \
                 [RealParameter(f"w{i + 1}", 0.0, 1.0) for i in range(dm)]
    else:
        levers = [RealParameter(f"c{i + 1}", -1.0, 1.0) for i in range(lm)] + \
                 [RealParameter(f"r{i + 1}", 0.0001, 1.0) for i in range(lm)] + \
                 [RealParameter(f"w{i + 1}", 0.0, 1.0) for i in range(dm)]

    constants = [
        Constant("mode", mode),
        Constant("graph", graph),
        Constant("police_start", police_start),
        Constant("n_realizations", n_realizations),
        Constant("t_max", t_max),
        Constant("sensor_location", sensor_location),
        Constant("sensor_detections", sensor_detections),
        Constant("fugitive_routes", fugitive_routes_labeled),
        Constant("rbf", rbf_cubic),
        Constant("labels_perunit_sorted", labels_perunit_sorted),
        Constant("labels_perunit_inv_sorted", labels_perunit_inv_sorted),
        Constant("labels_full_sorted", labels_full_sorted),
        Constant("labels_full_sorted_inv", labels_full_sorted_inv)
    ]

    if mode == 'optimization':
        outcomes = [
            ScalarOutcome('not_intercepted', kind=ScalarOutcome.MINIMIZE)
        ]

    elif mode == 'simulation':
        outcomes = [
            ArrayOutcome('at'),
            ArrayOutcome('police_paths')
        ]

    return levers, constants, outcomes
