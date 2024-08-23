from __future__ import division

import networkx as nx
import numpy as np
import pandas as pd
import random
import pickle
import math
from random import sample

import logging
logger = logging.getLogger(__name__)


class FugitiveInterception():
    def __init__(self, T, U, R, graph, units_start, nodesdict_perunit_sorted, fugitive_start, num_sensors, sensor_locations, fugitive_routes_db, multiobj=False, seed=1):
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

        self.fugitive_routes_db = fugitive_routes_db

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

    def f(self, P, mode='optimization'):
        T = self.T
        U = self.U
        R = self.R

        unit_routes_final = {f'route{r}': {f'unit{u}': [self.units_start[u]] for u in range(U)} for r in range(R)}
        unit_targetnodes_final = {f'route{r}': {f'unit{u}': [self.units_start[u]] for u in range(U)} for r in range(R)}
        policies = [None]

        fugitive_routes = random.sample(self.fugitive_routes_db, R)  # sample without replacement

        for r in range(R):
            sensor_detections = {}
            for sensor, location in enumerate(self.sensor_locations):
                sensor_detections['sensor' + str(sensor)] = list(map(int, [x in [location] for x in fugitive_routes[r]]))

            unit_route = {f'unit{u}': [self.units_start[u]] for u in range(U)}
            units_current = {f'unit{u}': self.units_start[u] for u in range(U)}
            # units_plan initially: shortest path to starting location of fugitive
            #units_plan = {f'unit{u}': nx.shortest_path(G=self.graph, source=self.units_start[u], target=self.fugitive_start) for u in range(U)}

            #units_plan initially: stay where they started
            units_plan = {f'unit{u}': [self.units_start[u]] for u in range(U)}
            units_targetnodes = {f'unit{u}': [self.units_start[u]] for u in range(U)}

            for u in range(U):
                if len(units_plan[f'unit{u}']) > 1:
                    del units_plan[f'unit{u}'][0]  # del current node from plan

            policies = [None]

            for t in range(1, T):
                # determine action from policy tree P based on indicator states
                #action, rules = P.evaluate(states=[t] + [sensor_detections['sensor' + str(s)][t] for s in range(self.num_sensors)])  #detection at t #TODO is it t or t-1?
                action, rules = P.evaluate(states=[t] + [int(any(sensor_detections['sensor' + str(s)][:t])) for s in range(
                    self.num_sensors)])  # states = list of current values of" ['Minute', 'SensorA']  # detection up until t

                # evaluate state transition function, given the action from the tree
                # the state transition function gives the next node for each of the units, given the current node and the action

                # 1) update planned paths
                unit_affected, _, target_node = action.split('_')
                #unit_affected_nr = int(list(unit_affected)[-1])
                units_targetnodes[unit_affected] = self.nodesdict_perunit_sorted[unit_affected][target_node]
                try:
                    if units_plan[unit_affected][-1] != target_node:
                        if nx.has_path(G=self.graph, source=units_current[unit_affected], target=self.nodesdict_perunit_sorted[unit_affected][target_node]):
                            units_plan[unit_affected] = nx.shortest_path(G=self.graph, source=units_current[unit_affected], target=self.nodesdict_perunit_sorted[unit_affected][target_node])
                        del units_plan[unit_affected][0]  # is source node (= current node)
                except IndexError:
                    if nx.has_path(G=self.graph, source=units_current[unit_affected], target=self.nodesdict_perunit_sorted[unit_affected][target_node]):
                        units_plan[unit_affected] = nx.shortest_path(G=self.graph, source=units_current[unit_affected],
                                                                     target=self.nodesdict_perunit_sorted[unit_affected][target_node])
                    if len(units_plan[unit_affected]) > 1:
                        del units_plan[unit_affected][0]  # is source node (= current node), but only if source =/= target

                # 2) update current position
                for u in range(U):
                    if len(units_plan[f'unit{u}']) > 0:
                        units_current[f'unit{u}'] = units_plan[f'unit{u}'][0]
                # 3) delete first entry from path list;
                for u in range(U):
                    if len(units_plan[f'unit{u}']) > 0:
                        del units_plan[f'unit{u}'][0]
                # 4) log past positions
                for u in range(U):
                    unit_route[f'unit{u}'].append(units_current[f'unit{u}'])

                # for rollout / visualization
                if mode == 'simulation':
                    policies.append(action)

            for u in range(U):
                unit_routes_final[f'route{r}'][f'unit{u}'] = unit_route[f'unit{u}']
                unit_targetnodes_final[f'route{r}'][f'unit{u}'] = units_targetnodes[f'unit{u}']

        df = pd.DataFrame()
        if mode == 'simulation':
            df['policy'] = pd.Series(policies, dtype='category')
            for r in range(R):
                df[f'fugitive_route{r}'] = pd.Series(fugitive_routes[r])
                for u in range(U):
                    df[f'fugitive_route{r}_unit{u}'] = pd.Series(unit_routes_final[f'route{r}'][f'unit{u}'])

            interception_dict = {}
            for r in range(R):
                interception_dict_perroute = {}
                for u in range(U):
                    interception_dict_perroute[f'unit{u}'] = [i for i, j in zip(unit_routes_final[f'route{r}'][f'unit{u}'], fugitive_routes[r])
                                                              if ((i == j) and (i == unit_targetnodes_final[f'route{r}'][f'unit{u}']))]  # intercept at target

                interception_dict[f'route{r}'] = any(len(value) for value in interception_dict_perroute.values())

            return df, interception_dict

        if mode == 'optimization':
            # objective function: calculate overlap with predicted escape routes
            interception_pct = 0
            for r in range(R):
                interception_dict = {}
                for u in range(U):
                    interception_dict[f'unit{u}'] = [i for i, j in zip(unit_routes_final[f'route{r}'][f'unit{u}'], fugitive_routes[r])
                                                     if ((i == j) and (i == unit_targetnodes_final[f'route{r}'][f'unit{u}']))]  # intercept at target

                # return objective value
                if any(len(value) for value in interception_dict.values()):
                    interception_pct += 1

            if not self.multiobj:
                return (R-interception_pct)/R  # minimize prob of interception

            else:
                interception_pct_final = (R - interception_pct) / R
                return [interception_pct_final, P.L.__len__()]  # multiobj 2: prob of intercept & number of nodes



