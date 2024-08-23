import networkx as nx
import pandas as pd
import numpy as np
import random

from network import test_graph as test_graph


def recalc_fug_routes(fugitive_routes_labeled, sensor_locations_labeled, sensor_triggered, time_step):
    """
    Filters fugitive routes based on sensor triggers
    """

    # for each fugitive route, would each sensor be triggered
    # sensor_detections = np.zeros((len(sensor_triggered), len(fugitive_routes_labeled)))
    # fugitive_routes_filtered = np.empty((0, len(fugitive_routes_labeled[0])))
    # for sensor, _ in enumerate(sensor_detections):
    #     sensor_detections[sensor, :] = np.where(fugitive_routes_labeled[:, time_step] == sensor_locations_labeled[sensor], 1, 0)
    #
    # for i, fug_route in enumerate(fugitive_routes_labeled):
    #     fug_route_can_be_true = False
    #     for sensor, _ in enumerate(sensor_detections):
    #         if sensor_detections[sensor, i] == sensor_triggered[sensor]:
    #             fug_route_can_be_true = True
    #
    #     if fug_route_can_be_true:
    #         fugitive_routes_filtered = np.vstack([fugitive_routes_filtered, fug_route])


    # fugitive_routes_filtered = np.empty((0, len(fugitive_routes_labeled[0])))
    fugitive_routes_filtered = []

    # is any sensor triggered?
    if any(sensor_triggered):

        for r, fug_route in enumerate(fugitive_routes_labeled):
            fug_route_can_be_true = False
            for sensor, triggered in enumerate(sensor_triggered):
                # if (fug_route[time_step] == sensor_locations_labeled[sensor]) == triggered:  # this works for neg triggers too
                try:
                    if (fug_route[time_step] == sensor_locations_labeled[sensor]) & triggered:
                        fug_route_can_be_true = True
                except IndexError:
                    pass

            if fug_route_can_be_true:
                # fugitive_routes_filtered = np.vstack([fugitive_routes_filtered, fug_route])
                fugitive_routes_filtered.append(fug_route)

    else:
        for r, fug_route in enumerate(fugitive_routes_labeled):
            fug_route_can_be_true = True
            for sensor, triggered in enumerate(sensor_triggered):
                try:
                    if (fug_route[time_step] == sensor_locations_labeled[sensor]) != triggered:
                        fug_route_can_be_true = False
                        break
                except IndexError:
                    pass

            if fug_route_can_be_true:
                # fugitive_routes_filtered = np.vstack([fugitive_routes_filtered, fug_route])
                fugitive_routes_filtered.append(fug_route)

    return fugitive_routes_filtered


def recalc_fug_routes_city(fugitive_routes_labeled, sensor_locations_labeled, sensor_triggered, time_interval):
    """
    Filters fugitive routes based on sensor triggers
    """

    fugitive_routes_filtered = []

    # is any sensor triggered?
    if any(sensor_triggered):

        for r, fug_route in enumerate(fugitive_routes_labeled):
            fug_route_can_be_true = False
            for sensor, triggered in enumerate(sensor_triggered):
                # if the route should trigger the sensor
                if sensor_locations_labeled[sensor] in [node for t, node in fug_route.items() if time_interval[0] <= t <= time_interval[1]]:
                    # and the sensor is triggered
                    if triggered:
                        # this route can be the ground truth
                        fug_route_can_be_true = True
                        break

            if fug_route_can_be_true:
                fugitive_routes_filtered.append(fug_route)

    else:
        for r, fug_route in enumerate(fugitive_routes_labeled):
            fug_route_can_be_true = True
            for sensor, triggered in enumerate(sensor_triggered):
                # if the route should trigger the sensor
                if sensor_locations_labeled[sensor] in [node for t, node in fug_route.items() if time_interval[0] <= t <= time_interval[1]]:
                    # but the sensor is not triggered
                    if triggered == False:
                        # this route cannot be the ground truth
                        fug_route_can_be_true = False
                        break

            if fug_route_can_be_true:
                fugitive_routes_filtered.append(fug_route)

    return fugitive_routes_filtered


if __name__ == '__main__':
    n_realizations = 10
    t_max = 5
    graph_type = 'test_graph_3'

    if graph_type == 'test_graph_2':
        graph, labels, labels_inv, pos = test_graph(n_paths=2)
    if graph_type == 'test_graph_3':
        graph, labels, labels_inv, pos = test_graph(n_paths=3)

    num_units = 1
    sensor_locations = [(1, 1)]
    sensor_locations_labeled = [labels[sensor] for sensor in sensor_locations]

    police_start = pd.read_pickle(
        f"../data/{graph_type}/units_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")
    fugitive_start = pd.read_pickle(
        f"../data/{graph_type}/fugitive_start_T{t_max}_R{n_realizations}_U{num_units}.pkl")
    # fugitive_routes = pd.read_pickle(
    #     f"../data/{graph_type}/fugitive_routes_T{t_max}_R{n_realizations}_U{num_units}.pkl")

    fugitive_routes = ([[(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)]] * 1 +
                       [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]] * 1 +
                       [[(0, 1), (1, 2), (2, 2), (3, 2), (4, 2)]] * 1)
    n_realizations = len(fugitive_routes)
    fugitive_routes_labeled = []
    for realization in range(n_realizations):
        list = []
        for t in range(t_max):
            list.append(labels[fugitive_routes[realization][t]])
        fugitive_routes_labeled.append(list)
    fugitive_routes_labeled = np.array(fugitive_routes_labeled)
    print(fugitive_routes_labeled)

    # idx = random.randint(0, len(fugitive_routes_labeled))
    idx = 1
    ground_truth_route = fugitive_routes[idx]
    ground_truth_route_labeled = fugitive_routes_labeled[idx]

    time_step = 1
    sensor_triggered = [sensor == ground_truth_route_labeled[time_step] for sensor in sensor_locations_labeled]  # based on 'ground truth' fugitive route

    # filter routes based on sensor info
    fugitive_routes_labeled = recalc_fug_routes(fugitive_routes_labeled, sensor_locations_labeled, sensor_triggered, time_step)

    print(fugitive_routes_labeled)
