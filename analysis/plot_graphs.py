import pickle
from time import gmtime, strftime
import osmnx as ox
import logging
from datetime import datetime
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def draw_edges(graph):
    edges_fugitive = []

    # for i_r, route_time in enumerate(fugitive_routes):
    #     route = list(route_time.values())
    #     for i, node in enumerate(route):
    #         if i ==0:
    #             continue
    #         else:
    #             edges_fugitive1 = [(route[i], route[i-1])]
    #             edges_fugitive2 = [(route[i-1], route[i])]
    #             edges_fugitive.extend(tuple(edges_fugitive1))
    #             edges_fugitive.extend(tuple(edges_fugitive2))

    edge_colormap = ['silver'] * len(graph.edges())
    edge_weightmap = [1] * len(graph.edges())
    for index, edge in enumerate(graph.edges()):
        if edge in edges_fugitive:
            edge_colormap[index] = 'tab:orange'
            edge_weightmap[index] = 2

    return edge_colormap, edge_weightmap


def draw_nodes(G, fugitive_start, escape_nodes, police_start, sensors):
    node_size = []
    node_color = []
    for node in G.nodes:
        # if node in police_end:
        #     node_size.append(120)
        #     node_color.append('tab:blue')
        if node in escape_nodes:
            node_size.append(20)
            node_color.append('tab:red')
        elif node in police_start:
            node_size.append(60)
            node_color.append('tab:blue')
        elif node in sensors:
            node_size.append(60)
            node_color.append('tab:green')
        elif node == fugitive_start:
            node_size.append(40)
            node_color.append('tab:orange')
        else:
            node_size.append(0)
            node_color.append('lightgray')
    return node_size, node_color


# def show_graph(G, escape_nodes, fugitive_start, save=False):
#     # filepath=f"graphs/FLEE/Graph_FLEE.graph.graphml"
#     # G = ox.load_graphml(filepath=filepath)
#
#     edge_colormap, edge_weightmap = draw_edges(G)
#     node_size, node_color = draw_nodes(G, fugitive_start, escape_nodes, police_start, police_end)
#
#     fig, ax = ox.plot_graph(
#         G, bgcolor="white", node_color=node_color, node_size=node_size, edge_linewidth=edge_weightmap,
#         edge_color=edge_colormap,
#     )
#     if save:
#         ax.savefig(f'graphs/{city}.png')


def plot_routes(city, num_units, num_sensors, instance):
    filepath = f"../data/city/graphs/{city}.graph.graphml"
    G = ox.load_graphml(filepath=filepath)

    with open(f'../data/city/graphs/escape_nodes_{city}.pkl', 'rb') as f:
        escape_nodes = pickle.load(f)
    with open(f'../data/city/sp_fugitive_start_{city}_T1800_R100_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl', 'rb') as f:
        fugitive_start = pickle.load(f)

    # get police start
    with open(f'../data/city/sp_units_start_{city}_T1800_R100_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl', 'rb') as f:
        police_start = pickle.load(f)
    # get sensor locs
    with open(f'../data/city/sp_sensors_{city}_T1800_R100_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl', 'rb') as f:
        sensors = pickle.load(f)

    # get fug routes
    with open(f'../data/city/sp_fugitive_routes_{city}_T1800_R100_U{num_units}_numsensors{num_sensors}_instance{instance}.pkl', 'rb') as f:
        fugitive_routes = pickle.load(f)

    fugitive_routes = [list(route.values()) for route in fugitive_routes]
    # results_routes += police_routes

    route_colors = ['tab:red' for r in fugitive_routes]
    route_alphas = [0.05 for r in fugitive_routes]
    route_linewidths = [1 for r in fugitive_routes]

    # nx.draw_networkx_edges(G,edgelist=path_edges,edge_color='r',width=10)
    node_size, node_color = draw_nodes(G, fugitive_start, escape_nodes, police_start, sensors)
    edge_colormap, edge_weightmap = draw_edges(G)
    edge_weightmap = [0.3] * len(G.edges())

    fig, ax = ox.plot_graph_routes(G, fugitive_routes,
                                   route_linewidths=route_linewidths, route_alpha=0.05, route_colors=route_colors,
                                   edge_linewidth=edge_weightmap, edge_color=edge_colormap,
                                   node_color=node_color, node_size=node_size, node_zorder=2,
                                   bgcolor="white",
                                   orig_dest_size=30,
                                   show=False,
                                   # orig_dest_node_color=['tab:orange', 'tab:red']*len(results_routes),
                                   )

    fig.savefig(f'figs/maps/{city}_U{num_units}_S{num_sensors}_instance{instance}.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':

    # for city in ['Manhattan']:
    for city in ['Winterswijk', 'Utrecht', 'Manhattan']:
        for num_units in [3, 10]:
            for num_sensors in [3, 10]:
                for instance in range(10):
                    plot_routes(city, num_units, num_sensors, instance)
                    print(datetime.now().strftime("%H:%M:%S"), 'done: ', city, num_units, num_sensors, instance)

