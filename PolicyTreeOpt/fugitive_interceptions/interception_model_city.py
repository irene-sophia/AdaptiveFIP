from __future__ import division

import networkx as nx
import numpy as np
import pandas as pd
import random
import pickle
import math
from random import sample
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


class FugitiveInterception():
    def __init__(self, T, U, R, graph, units_start, nodesdict_perunit_sorted, fugitive_start, num_sensors, sensor_locations, fugitive_routes_db, fugitive_routes, multiobj=False, seed=1):
        self.T = T
        self.U = U
        self.R = R
        self.graph = graph
        self.units_start = units_start
        self.nodesdict_perunit_sorted = nodesdict_perunit_sorted
        self.fugitive_start = fugitive_start
        self.num_sensors = num_sensors
        self.sensor_locations = sensor_locations
        self.multiobj = multiobj
        self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)

        logger.info("sampled police starts: {}".format(units_start))
        logger.info("sampled fugitive start: {}".format(fugitive_start))

        # simulate fugitive escape routes
        # fugitive_routes_db = []
        # for r in range(R):
        #     route = self.escape_route()
        #     fugitive_routes_db.append(route)

        self.fugitive_routes = fugitive_routes_db
        self.fugitive_routes_labeled = fugitive_routes

    def escape_route(self):
        walk = []
        node = self.fugitive_start
        walk.append(node)

        for i in range(self.T - 1):
            list_neighbor = list(self.graph.neighbors(node))

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

    def f(self, trees, mode='optimization'):
        T = self.T
        max_timestep = int(T/30)
        U = self.U
        R = self.R

        unit_routes_final = {f'route{r}': {f'unit{u}': [self.units_start[u]] for u in range(U)} for r in range(R)}
        unit_targetnodes_final = {f'route{r}': {f'unit{u}': {0: self.units_start[u]} for u in range(U)} for r in range(R)}
        policies = [None]

        # fugitive_routes = random.sample(self.fugitive_routes_db, R)  # sample without replacement
        fugitive_routes = self.fugitive_routes_labeled.copy()

        for r in range(R):
        # for r in range(2):
            sensor_detections = {}
            for sensor, location in enumerate(self.sensor_locations):
                time_intervals = [[(t-1)*30, t*30] for t in range(1, max_timestep+1)]
                sensor_detections['sensor' + str(sensor)] = [location in [node for t, node in fugitive_routes[r].items()
                                       if time_intervals[time_step][0] <= t <= time_intervals[time_step][1]] for time_step in range(max_timestep)]

            # unit_route = {f'unit{u}': [self.units_start[u]] for u in range(U)}
            unit_route = {f'unit{u}': {0: self.units_start[u]} for u in range(U)}
            units_current = {f'unit{u}': self.units_start[u] for u in range(U)}
            # units_plan initially: shortest path to starting location of fugitive
            #units_plan = {f'unit{u}': nx.shortest_path(G=self.graph, source=self.units_start[u], target=self.fugitive_start) for u in range(U)}
            units_progress_on_link = {u: 0 for u in range(U)}

            #units_plan initially: stay where they started
            units_plan = {f'unit{u}': [self.units_start[u]] for u in range(U)}
            units_targetnodes = {f'unit{u}': {0: self.units_start[u]} for u in range(U)}

            # for u in range(U):
            #     if len(units_plan[f'unit{u}']) > 1:
            #         del units_plan[f'unit{u}'][0]  # del current node from plan

            policies = [None]

            for t in range(max_timestep+1):
                if t == 0:
                    time_interval = [0, 0]
                else:
                    time_interval = [(t - 1) * 30, t * 30]

                #eval tree for each unit
                for u in range(U):
                    P=trees[u]
                    # determine action from policy tree P based on indicator states
                    #action, rules = P.evaluate(states=[t] + [sensor_detections['sensor' + str(s)][t] for s in range(self.num_sensors)])  #detection at t
                    action, rules = P.evaluate(states=[t] + [int(any(sensor_detections['sensor' + str(s)][:t])) for s in range(
                        self.num_sensors)])  # states = list of current values of" ['Minute', 'SensorA']  # detection up until t

                    # evaluate state transition function, given the action from the tree
                    # the state transition function gives the next node for each of the units, given the current node and the action

                    # 1) update planned paths
                    # unit_affected is always u (dep on arguments passed to each tree)
                    unit_affected, _, target_node = action.split('_')
                    unit_affected = f'unit{u}'
                    #unit_affected_nr = int(list(unit_affected)[-1])
                    units_targetnodes[unit_affected][t] = self.nodesdict_perunit_sorted[unit_affected][target_node]
                    try:
                        if units_plan[unit_affected][-1] != self.nodesdict_perunit_sorted[unit_affected][target_node]:
                            if nx.has_path(G=self.graph, source=units_current[unit_affected], target=self.nodesdict_perunit_sorted[unit_affected][target_node]):
                                units_plan[unit_affected] = nx.shortest_path(G=self.graph, source=units_current[unit_affected], target=self.nodesdict_perunit_sorted[unit_affected][target_node], weight='travel_time')
                                if len(units_plan[unit_affected]) > 1:
                                    del units_plan[unit_affected][0]  # is source node (= current node)
                                # elif len(units_plan[unit_affected]) <= 1:
                                #     pass

                    except IndexError:  # no units_plan yet
                        if nx.has_path(G=self.graph, source=units_current[unit_affected], target=self.nodesdict_perunit_sorted[unit_affected][target_node]):
                            units_plan[unit_affected] = nx.shortest_path(G=self.graph, source=units_current[unit_affected],
                                                                         target=self.nodesdict_perunit_sorted[unit_affected][target_node], weight='travel_time')
                            if len(units_plan[unit_affected]) > 1:
                                del units_plan[unit_affected][0]  # is source node (= current node), but only if source =/= target

                    # for u in range(U):
                    # make moves for this time interval for this unit
                    try:
                        _t = time_interval[0]

                        if (units_progress_on_link[u] != 0) and (units_current[f'unit{u}'] != units_plan[f'unit{u}'][-1]):
                            # calculate time left on link
                            link_travel_time_left = min(
                                [float(self.graph[units_current[f'unit{u}']][units_plan[f'unit{u}'][0]][i]['travel_time']) for i in
                                 self.graph[units_current[f'unit{u}']][units_plan[f'unit{u}'][0]]]) - units_progress_on_link[u]

                            if link_travel_time_left > 30:  # time step interval
                                units_progress_on_link[u] += 30
                                _t += 31
                            else:
                                # add that time to t
                                _t += link_travel_time_left
                                units_progress_on_link[u] = 0

                                units_current[f'unit{u}'] = units_plan[f'unit{u}'][0]
                                if len(units_plan[f'unit{u}']) > 1:
                                    del units_plan[f'unit{u}'][0]
                                # unit_route[f'unit{u}'].append(units_current[f'unit{u}'])
                                unit_route[f'unit{u}'][_t] = units_current[f'unit{u}']

                        # try:
                        # while time left in interval and unit not arrived at current target node
                        # don't do target note bc if there is no path, this is 'while True'
                        # while _t <= time_interval[1] and units_current[f'unit{u}'] != units_targetnodes['unit0'][max(units_targetnodes['unit0'])]:
                        while (_t <= time_interval[1]) and (units_current[f'unit{u}'] != units_plan[f'unit{u}'][-1]):
                            # go towards units plan if t permits:
                            _t += min([float(self.graph[units_current[f'unit{u}']][units_plan[f'unit{u}'][0]][i]['travel_time'])
                                 for i in self.graph[units_current[f'unit{u}']][units_plan[f'unit{u}'][0]]])
                            # _t += min([float(self.graph[units_current[f'unit{u}']][units_plan[f'unit{u}'][0]][i]['travel_time'])
                            #      for i in self.graph[units_current[f'unit{u}']][units_plan[f'unit{u}'][0]]]) - units_progress_on_link[u]

                            # if cannot reach next node in time, update progress on node
                            if _t > time_interval[1]:
                                units_progress_on_link[u] = _t - time_interval[1]

                            else:
                                units_current[f'unit{u}'] = units_plan[f'unit{u}'][0]
                                unit_route[f'unit{u}'][_t] = units_current[f'unit{u}']

                                if len(units_plan[f'unit{u}']) > 1:  # otherwise has arrived at target and while loop will stop
                                    ### update current position
                                    # units_current[f'unit{u}'] = units_plan[f'unit{u}'][0]
                                    ### delete first entry from units_plan
                                    del units_plan[f'unit{u}'][0]
                                    ### log position in unit_route
                                    # unit_route[f'unit{u}'].append(units_current[f'unit{u}'])
                                    # unit_route[f'unit{u}'][_t] = units_current[f'unit{u}']

                                ### if simulation: policies.append(action)
                                if mode == 'simulation':
                                    policies.append(action)
                    except:
                        pass

            for u in range(U):
                # unit_routes_final[f'route{r}'][f'unit{u}'] = unit_route[f'unit{u}']
                unit_routes_final[f'route{r}'] = unit_route
                unit_targetnodes_final[f'route{r}'] = units_targetnodes

        police_at_target = defaultdict(lambda: defaultdict(dict))
        for realization, unit_routes_ in unit_routes_final.items():
            for u, route in unit_routes_.items():
                for t in route.keys():
                    associated_timestep = int(np.floor(t / 30))
                    try:
                        if route[t] == unit_targetnodes_final[realization][u][associated_timestep]:
                            if route[t] not in police_at_target[realization][u].keys():
                                if t == max(route):  # last entry
                                    police_at_target[realization][u][route[t]] = [
                                        {'arrival_time': t, 'departure_time': max_timestep * 31}]
                                else:
                                    dep_time = list(route)[list(route).index(t) + 1]
                                    police_at_target[realization][u][route[t]] = [
                                        {'arrival_time': t, 'departure_time': dep_time}]
                            else:
                                # if t == max(route):  # last entry
                                if t == list(route)[-1]:  # TODO: still sometimes some higher val is inserted in dict first ^see error if using above line
                                    police_at_target[realization][u][route[t]].append(
                                        {'arrival_time': t, 'departure_time': max_timestep * 31})
                                else:
                                    dep_time = list(route)[list(route).index(t) + 1]
                                    police_at_target[realization][u][route[t]].append(
                                        {'arrival_time': t, 'departure_time': dep_time})
                    except:
                        pass

        df = pd.DataFrame()
        if mode == 'simulation':
            df['policy'] = pd.Series(policies, dtype='category')
            for r in range(R):
                df[f'fugitive_route{r}'] = pd.Series(fugitive_routes[r])
                for u in range(U):
                    df[f'fugitive_route{r}_unit{u}'] = pd.Series(unit_routes_final[f'route{r}'][f'unit{u}'])

            # check for interception
            interceptions = {}
            for r, route in enumerate(fugitive_routes):
                interceptions[r] = 0

            for r, (realization, u_nodes_pol) in enumerate(police_at_target.items()):
                for u, nodes_pol in u_nodes_pol.items():
                    for fug_time_on_node, node_fug in fugitive_routes[r].items():
                        if node_fug in nodes_pol.keys():
                            for period in nodes_pol[node_fug]:
                                if period['arrival_time'] <= fug_time_on_node <= period['departure_time']:
                                    interceptions[realization] = 1

            return df, interceptions

        if mode == 'optimization':
            # check for interception
            interceptions = {}
            for r, route in enumerate(fugitive_routes):
                interceptions[r] = 0

            for r, (realization, u_nodes_pol) in enumerate(police_at_target.items()):
                for u, nodes_pol in u_nodes_pol.items():
                    for fug_time_on_node, node_fug in fugitive_routes[r].items():
                        if node_fug in nodes_pol.keys():
                            for period in nodes_pol[node_fug]:
                                if period['arrival_time'] <= fug_time_on_node <= period['departure_time']:
                                    interceptions[r] = 1

            interception_pct = sum(interceptions.values())

            if not self.multiobj:
                return (R-interception_pct)/R  # minimize prob of interception

            else:
                interception_pct_final = (R - interception_pct) / R
                # TODO make that work for the perunit mode
                return [interception_pct_final, P.L.__len__()]  # multiobj 2: prob of intercept & number of nodes



