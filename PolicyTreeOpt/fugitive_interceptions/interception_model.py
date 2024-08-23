from __future__ import division

import networkx as nx
import numpy as np
import pandas as pd
import random
from random import sample


class FugitiveInterception():
    def __init__(self, T, U, graph, units_start, fugitive_start, num_sensors, sensor_locations):
        self.T = T
        self.U = U
        self.graph = graph
        self.units_start = units_start
        self.fugitive_start = fugitive_start
        self.units_current = {f'unit{u}': units_start[u] for u in range(U)}
        self.num_sensors = num_sensors
        self.sensor_locations = sensor_locations

        units_plan = {f'unit{u}': nx.shortest_path(G=graph, source=units_start[u], target=fugitive_start) for u in range(U)}
        self.units_plan = units_plan

    def escape_route(self):
        walk = []
        node = self.fugitive_start
        walk.append(node)

        for i in range(self.T):
            list_neighbor = list(self.graph.neighbors(node))

            if i == 0:
                previous_node = node
                nextnode = random.choice(list_neighbor)

            else:
                # exclude previous node for 'normal' walk
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
        units_plan = self.units_plan
        units_current = self.units_current
        unit_routes_final = {f'unit{u}': [self.units_start[u]] for u in range(U)}
        policies = [None]

        # simulate fugitive escape route (different each function evaluation!)
        fugitive_route = self.escape_route()

        sensor_detections = {}
        for sensor, location in enumerate(self.sensor_locations):
            sensor_detections['sensor'+str(sensor)] = list(map(int, [x in [location] for x in fugitive_route]))

        for t in range(1, T):
            # determine action from policy tree P based on indicator states
            action, rules = P.evaluate(states=[t] + [sensor_detections['sensor'+str(s)][t] for s in range(self.num_sensors)]) #states = list of current values of" ['Minute', 'SensorA']

            # evaluate state transition function, given the action from the tree
            # the state transition function gives the next node for each of the units, given the current node and the action

            # dictionary to translate nodes to node numbers
            i = 0
            nodes_dict = {}
            for node in self.graph.nodes():
                nodes_dict['node'+str(i)] = node
                i += 1

            #1) update planned paths
            unit_affected, _, target_node = action.split('_')
            try:
                if units_plan[unit_affected][-1] != target_node:
                    units_plan[unit_affected] = nx.shortest_path(G=self.graph, source=units_current[unit_affected], target=nodes_dict[target_node])
            except IndexError:
                units_plan[unit_affected] = nx.shortest_path(G=self.graph, source=units_current[unit_affected],
                                                             target=nodes_dict[target_node])

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
                unit_routes_final[f'unit{u}'].append(units_current[f'unit{u}'])

            # for rollout / visualization
            if mode == 'simulation':
                policies.append(action)

        df = pd.DataFrame()
        if mode == 'simulation':
            df['policy'] = pd.Series(policies, dtype='category')
            df['fugitive_route'] = pd.Series(fugitive_route)
            for u in range(U):
                df[f'unit{u}'] = pd.Series(unit_routes_final[f'unit{u}'])

            interception_dict = {}
            for u in range(U):
                interception_dict[f'unit{u}'] = [i for i, j in
                                                      zip(unit_routes_final[f'unit{u}'], fugitive_route) if
                                                      i == j]

            return df, any(len(value) for value in interception_dict.values())

        if mode == 'optimization':
            # objective function: calculate overlap with predicted escape routes
            interception_dict = {}
            for u in range(U):
                interception_dict[f'unit{u}'] = [i for i, j in zip(unit_routes_final[f'unit{u}'], fugitive_route) if i == j]
            # return objective value
            if any(len(value) for value in interception_dict.values()):
                return 0  # maximize interception
            else:
                return 1
