import math
import numpy as np
import networkx as nx
import pandas as pd
import pickle

# from DPS.rbf import rbf_cubic
from collections import defaultdict
import itertools
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def fugitive_interception_model(
        graph,
        police_start,
        n_realizations=3,
        t_max=5,
        sensor_location=None,
        fugitive_routes=np.zeros((3, 5)),
        rbf=None, num_rbfs=1,
        labels_perunit_sorted=None,
        labels_perunit_inv_sorted=None,
        labels_full_sorted=None,
        labels_full_sorted_inv=None,
        sensor_detections=None,
        mode='optimization',
        **kwargs
):
    """
    Runs the fugitive interception model for N realizations and
    returns the fraction of realizations where the fugitive is intercepted

    Parameters
    -------

    Returns
    -------
    scalar: fraction of intercepted routes
    """

    c_i = []
    r_i = []
    w_i = []
    counter = 1
    for _ in range(num_rbfs):
        for _ in range(len(sensor_location)):
            c_i.append(kwargs[f'c{counter}'])
            r_i.append(kwargs[f'r{counter}'])
            counter += 1

    counter = 1
    for _ in range(num_rbfs):
        for _ in range(len(police_start)):
            w_i.append(kwargs[f'w{counter}'])  # weight
            counter += 1

    c_i = np.asarray(c_i)
    r_i = np.asarray(r_i)
    w_i = np.asarray(w_i)

    centers = c_i.reshape((len(sensor_location), num_rbfs))
    radii = r_i.reshape((len(sensor_location), num_rbfs))
    weights = w_i.reshape((len(police_start), num_rbfs))

    # police_path: output dict of simulation time: node for each unit for each realization {realization: {unit: []}}
    police_path = defaultdict(dict)
    # police planned path: dict of list of nodes representing the shortest path from police_current to target_node
    police_planned_path = {u: [police_start[u]] * n_realizations for u, _ in enumerate(police_start)}
    # police_progress_on_edge: array of time traveled on current edge
    police_progress_on_link = np.zeros((len(police_start), n_realizations))
    # police current: dict of position of each police unit at time step t
    police_current = {u: [police_start[u]] * n_realizations for u, _ in enumerate(police_start)}

    actions_df = pd.DataFrame()

    for timestep in range(int(t_max/30)):
        for u, _ in enumerate(police_start):
            # obtain actions (vector of actions per realization)
            at = sum(sum([rbf(xt=sensor_detections[l][:, timestep], c_lm=centers[l, m], r_lm=radii[l, m], w_md=weights[u, m],
                          labels_perunit_sorted=labels_perunit_sorted, u=u) for l, _ in enumerate(sensor_location)]) for m in range(num_rbfs))

            at = np.clip(at, 0, (len(labels_perunit_sorted[u]) - 0.0001))
            actions_df[str(u) + '_' + str(timestep)] = [labels_perunit_inv_sorted[u][i] for i in np.floor(at).astype(int)]

            # update police_planned_path = shortest path to at/target_node
            for realization in range(n_realizations):
                target_node = labels_perunit_inv_sorted[u][int(np.floor(at[realization]))]
                if police_progress_on_link[u, realization] != 0:
                    pass

                if timestep != 0:
                    prev_target_node = police_planned_path[u][realization][-1]

                if timestep == 0:
                    # 1a) set police_planned_path to shortest path to target_node
                    if nx.has_path(G=graph, source=labels_full_sorted_inv[police_current[u][realization]],
                                   target=target_node):
                        police_planned_path[u][realization] = nx.shortest_path(G=graph,
                                                                               source=labels_full_sorted_inv[
                                                                                   police_current[u][realization]],
                                                                               target=target_node)

                    # 2) remove current node from the planned path
                    if police_current[u][realization] != labels_full_sorted[target_node]:
                        del police_planned_path[u][realization][0]  # is source node (= current node)

                    # 3) if t== 0: add starting node to police_path
                    police_path[realization][u] = {0: police_current[u][realization]}

                elif target_node != prev_target_node:
                    # 1b) recalculate planned path
                    if police_progress_on_link[u, realization] == 0:
                        # if no progress towards next node, calculate path from current node
                        if nx.has_path(G=graph, source=labels_full_sorted_inv[police_current[u][realization]],
                                       target=target_node):
                            police_planned_path[u][realization] = nx.shortest_path(G=graph,
                                                                                   source=labels_full_sorted_inv[
                                                                                       police_current[u][realization]],
                                                                                   target=target_node)

                        # 2) remove current node from the planned path
                        if police_current[u][realization] != labels_full_sorted[target_node]:
                            del police_planned_path[u][realization][0]  # is source node (= current node)
                    else:
                        # if progress towards next node, calculate from next node and add the next node to the path
                        next_node = police_planned_path[u][realization][0]
                        if nx.has_path(G=graph, source=next_node, target=target_node):
                            police_planned_path[u][realization] = nx.shortest_path(G=graph,
                                                                                   source=next_node,
                                                                                   target=target_node)

                else:
                    # 1c) target_node != prev_target_node -> keep going as is
                    pass

                # 4) check if target_node == current node. if not, travel towards goal
                if target_node != labels_full_sorted_inv[police_current[u][realization]]:
                    time_prev_node = max(police_path[realization][u].keys())
                    time_next_node = time_prev_node + graph.edges[
                        labels_full_sorted_inv[police_current[u][realization]],
                        police_planned_path[u][realization][0], 0][
                        'travel_time']

                    # while time left in time step to reach next node, take step towards target node
                    while time_next_node <= (time_prev_node + police_progress_on_link[u, realization] + 30): # 1= timestep size
                        police_progress_on_link[u, realization] = 0
                        # update current node
                        police_current[u][realization] = labels_full_sorted[police_planned_path[u][realization][0]]
                        # add to police path
                        police_path[realization][u][time_next_node] = labels_full_sorted[police_planned_path[u][realization][0]]
                        # remove from planned path
                        if len(police_planned_path[u][realization]) > 1:
                            del police_planned_path[u][realization][0]
                        else:
                            break
                        # calc time to next node
                        time_next_node += graph.edges[
                            labels_full_sorted_inv[police_current[u][realization]],
                            police_planned_path[u][realization][0], 0]['travel_time']

                    else:  # stay on node, but update progress on link
                        police_progress_on_link[u, realization] = (timestep*30) - max(police_path[realization][u].keys())


    if mode == 'optimization':
        # construct police_at_target dict
        police_at_target = defaultdict(lambda: defaultdict(dict))
        for realization, unit_route in police_path.items():
            for u, route in unit_route.items():
                for t in route.keys():
                    if t == 0:
                        prev_t = 0
                    if labels_full_sorted[actions_df.iloc[realization, :][f'{u}_{int(np.floor((t/30)))}']] == route[t]:
                        police_at_target[realization][u][route[t]] = {'arrival_time': t, 'departure_time': t_max+1}

                    # if not on target now, but the actions changed compared to previous time step
                    elif actions_df.iloc[realization, :][f'{u}_{int(np.floor((t/30)))}'] != actions_df.iloc[realization, :][f'{u}_{max(0, int(np.floor((t/30)-1)))}']:
                        # if previous time step, the police node was the target node
                        if labels_full_sorted[actions_df.iloc[realization, :][f'{u}_{int(np.floor((prev_t/30)))}']] == route[prev_t]:
                            # but current time step, the police node and target nodes are different
                            if labels_full_sorted[actions_df.iloc[realization, :][f'{u}_{int(np.floor((t/30)))}']] != route[t]:
                                police_at_target[realization][u][route[prev_t]]['departure_time'] = t  # because of 'teleportation' discretization - otherwise prev_t

                    # set prev_t
                    prev_t = t

        # check for interception
        interceptions = {}
        for r, route in enumerate(fugitive_routes):
            interceptions[r] = 0

        for realization, u_nodes_pol in police_at_target.items():
            for u, nodes_pol in u_nodes_pol.items():
                for fug_time_on_node, node_fug in fugitive_routes[realization].items():
                    if node_fug in nodes_pol.keys():
                        if nodes_pol[node_fug]['arrival_time'] <= fug_time_on_node <= nodes_pol[node_fug]['departure_time']:
                            interceptions[realization] = 1

        num_intercepted = sum(interceptions.values())
        # print('percentage intercepted: ', num_intercepted/n_realizations)
        return {'not_intercepted': (n_realizations - num_intercepted) / n_realizations}

        # for r, route in enumerate(fugitive_routes):  # for each route
        #     if any([node in pi_nodes for node in r.values()]):
        #         for u, pi in enumerate(pi_nodes):  # for each police unit
        #             for time_at_node_fugitive, node_fugitive in r.items():  # for each node in the fugitive route
        #                 if node_fugitive == pi:  # if the fugitive node is the same as the target node of the police unit
        #                     if time_at_node_fugitive > tau_uv[
        #                         u, node_fugitive]:  # and the police unit can reach that node
        #                         z_r[i_r] = 1  # intercepted

        # interceptions_array = np.zeros((n_realizations, t_max))
        # for u, _ in enumerate(police_start):
        #     interceptions_array = interceptions_array + (police_path[u] == fugitive_routes)
        # # for each row, sum over bool (intercepted). sum over all realizations
        # num_intercepted = sum(interceptions_array.sum(axis=1) != 0)
        # # MINIMIZE number of routes NOT intercepted
        # return {'not_intercepted': (n_realizations - num_intercepted) / n_realizations}

    if mode == 'simulation':
        actions_df.to_csv(f'./results/actions_tt_{rbf.__name__}S{len(sensor_location)}_U{len(police_start)}.csv')

        pickle.dump(police_path, open(
            f'./results/police_path_tt_{rbf.__name__}_S{len(sensor_location)}_U{len(police_start)}.pkl',
            'wb'))

        return {'at': actions_df, 'police_paths': police_path}
