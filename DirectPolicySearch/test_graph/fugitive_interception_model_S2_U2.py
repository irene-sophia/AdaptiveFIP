import math
import numpy as np
import networkx as nx
import pandas as pd

from DPS.rbf import rbf_cubic


def fugitive_interception_model(
        graph,
        police_start,
        n_realizations=3,
        t_max=5,
        sensor_location=None,
        fugitive_routes=np.zeros((3, 5)),
        rbf=rbf_cubic,
        labels_perunit_sorted=None,
        labels_perunit_inv_sorted=None,
        labels_full_sorted=None,
        labels_full_sorted_inv=None,
        mode='optimization',
        c1=None,
        c2=None,

        r1=None,
        r2=None,

        w1=None,
        w2=None):
    """
    Runs the fugitive interception model for N realizations and
    returns the fraction of realizations where the fugitive is intercepted

    Parameters
    -------

    Returns
    -------
    scalar: fraction of intercepted routes
    """

    if len(sensor_location) == 1:
        centers = [c1]
        radii = [r1]
    elif len(sensor_location) == 2:
        centers = [c1, c2]
        radii = [r1, r2]

    if len(police_start) == 1:
        # police_start = police_start[0]  # until code is ready to handle multiple units
        weights = [w1]
    elif len(police_start) == 2:
        weights = [w1, w2]

    sensor_detections = {i: np.zeros((n_realizations, t_max)) for i, _ in enumerate(sensor_location)}
    police_path = {u: np.ones((n_realizations, t_max)) * 424242 for u, _ in enumerate(police_start)}
    police_planned_path = {u: [police_start[u]] * n_realizations for u, _ in enumerate(police_start)}
    police_current = {u: [police_start[u]] * n_realizations for u, _ in enumerate(police_start)}

    actions_df = pd.DataFrame()
    # for t in T_max:
    for t in range(t_max):

        # get xt (sensor_detections) from fugitive_routes and sensor locations

        # only signal when on sensor
        # sensor_detections[:, t] = np.where(fugitive_routes[:, t] == sensor_location, 1, 0)

        # keep signal on after sensing
        if t == 0:
            for i, sensor_loc in enumerate(sensor_location):
                sensor_detections[i][:, t] = np.where(fugitive_routes[:, t] == sensor_loc, 1, 0)
        else:
            for i, sensor_loc in enumerate(sensor_location):
                detections_till_tsensor = np.zeros(n_realizations, )
                for t_sensor in range(t + 1):
                    detections_till_tsensor = detections_till_tsensor + np.where(fugitive_routes[:, t_sensor] == sensor_loc, 1, 0)
                sensor_detections[i][:, t] = detections_till_tsensor

        for u, _ in enumerate(police_start):
            # evaluate rbf to obtain at (=target_node)
            at = sum([rbf(xt=sensor_detections[i][:, t], c_lm=centers[i], r_lm=radii[i], w_md=weights[u], labels_perunit_sorted=labels_perunit_sorted) for i, _ in enumerate(sensor_location)])
            at = np.clip(at, 0, (len(labels_perunit_sorted[u]) - 0.0001))
            actions_df[str(u) + '_' + str(t)] = [labels_perunit_inv_sorted[u][i] for i in np.floor(at).astype(int)]

            # update units_current (array shape (Nrealizations, Nunits) according to at (shortest route)
            for realization in range(n_realizations):
                # 1) police_planned_path = shortest path to at/target_node
                target_node = labels_perunit_inv_sorted[u][int(np.floor(at[realization]))]

                if nx.has_path(G=graph, source=labels_full_sorted_inv[police_current[u][realization]], target=target_node):
                    police_planned_path[u][realization] = nx.shortest_path(G=graph,
                                                                            source=labels_full_sorted_inv[police_current[u][realization]],
                                                                            target=target_node)

                # 2) remove current node from the planned path
                if police_current[u][realization] != labels_full_sorted[target_node]:
                    del police_planned_path[u][realization][0]  # is source node (= current node)

                # 3) update path history 'police_path'
                police_path[u][realization, t] = police_current[u][realization]

                # 4) update current position: take 1 step towards the target node
                police_current[u][realization] = labels_full_sorted[police_planned_path[u][realization][0]]


    if mode == 'optimization':
        # check for interception
        interceptions_array = np.zeros((n_realizations, t_max))
        for u, _ in enumerate(police_start):
            interceptions_array = interceptions_array + (police_path[u] == fugitive_routes)
        # for each row, sum over bool (intercepted). sum over all realizations
        num_intercepted = sum(interceptions_array.sum(axis=1) != 0)
        # MINIMIZE number of routes NOT intercepted
        return {'not_intercepted': (n_realizations - num_intercepted) / n_realizations}


    if mode == 'simulation':
        police_path_output = pd.DataFrame([[[labels_full_sorted_inv[i] for i in j] for j in np.floor(police_path[u]).astype(int)] for u,_ in enumerate(police_start)])
        actions_df.to_csv(f'./results/actions_S{len(sensor_location)}_U{len(police_start)}.csv')
        police_path_output.to_csv(f'./results/police_path_S{len(sensor_location)}_U{len(police_start)}.csv')
        return {'at': actions_df, 'police_paths': police_path_output}
