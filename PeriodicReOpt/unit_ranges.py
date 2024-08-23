import pandas as pd
import numpy as np
import networkx as nx
import random


def unit_ranges(start_units, U, G, labels_full_sorted):
    units_range_index = pd.MultiIndex.from_product(
        [range(U), range(len(labels_full_sorted))], names=("unit", "vertex")
    )
    units_range_time = pd.DataFrame(index=units_range_index, columns=["time_to_reach"])

    for u in range(U):
        for v in labels_full_sorted:
            if nx.has_path(G, start_units[u], v):
                units_range_time.loc[(u,labels_full_sorted[v])] = nx.shortest_path_length(G,
                                                                                          source=start_units[u],
                                                                                          target=v,
                                                                                          weight='travel_time',
                                                                                          method='bellman-ford')
            else:
                units_range_time.loc[(u,labels_full_sorted[v])] = 424242

    units_range_time = units_range_time.fillna(0)

    return np.squeeze(units_range_time).to_dict()
