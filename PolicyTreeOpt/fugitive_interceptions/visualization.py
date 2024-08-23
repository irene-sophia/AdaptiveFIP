import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# import imageio
import math
import glob
import os


def draw_edges(graph, results_df, step, T, U, route_to_vis):
    fugitive_route = results_df[f'fugitive_route{route_to_vis}'].tolist()

    edges_fugitive = []
    edges_units = []

    if step in range(1,T):
        # for i in range(1, step+1):  #to plot entire tail
        edges_fugitive1 = [(fugitive_route[step], fugitive_route[step-1])]
        edges_fugitive2 = [(fugitive_route[step-1], fugitive_route[step])]
        edges_fugitive.extend(tuple(edges_fugitive1))
        edges_fugitive.extend(tuple(edges_fugitive2))

        for u in range(U):
            unit_route = results_df[f'fugitive_route{route_to_vis}_unit{u}']
            edges_units.extend([(unit_route[step], unit_route[step-1])])
            edges_units.extend([(unit_route[step-1], unit_route[step])])

    edge_colormap = ['lightgray'] * len(graph.edges())
    edge_weightmap = [1] * len(graph.edges())
    for index, edge in enumerate(graph.edges()):
        # fugitive
        if edge in edges_fugitive:
            edge_colormap[index] = 'tab:orange'
            edge_weightmap[index] = 2
        # units
        elif edge in edges_units:
            edge_colormap[index] = 'tab:blue'
            edge_weightmap[index] = 2

    return edge_colormap, edge_weightmap


def draw_nodes(graph, results_df, sensor_locations, U, R, step, labels, pos, route_to_vis):
    node_colormap = ['lightgray']*len(graph.nodes)
    labels_at_step = labels.copy()

    sameplace_colormap = []
    sameplace_pos = dict()
    sameplace_graph = nx.Graph()

    for index, node in enumerate(graph.nodes()):
        # sensors
        if node in sensor_locations:
            node_colormap[index] = '#4caf50'
            #labels_at_step[node] = 'sensor'+str(sensor_locations.index(node))
        # fugitive route
        if node == results_df[f'fugitive_route{route_to_vis}'].tolist()[step]:
            if node_colormap[index] == 'tab:orange':
                node_colormap[index] = 'tab:orange'
            elif node_colormap[index] == 'lightgray':
                node_colormap[index] = 'tab:orange'
            else:  # blue (unit also at this node)
                sameplace_graph.add_node(node)
                sameplace_pos[node] = pos[node]
                if len(sameplace_graph.nodes) > len(sameplace_colormap):  # for if 3 at the same place
                    sameplace_colormap.append('tab:orange')
        # unit routes final
        for u in range(U):
            if node == results_df[f'fugitive_route{route_to_vis}_unit{u}'].tolist()[step]:
                if node_colormap[index] == 'lightgray':
                    node_colormap[index] = 'tab:blue'
                    #labels_at_step[node] = 'unit'+str(u)
                else:
                    sameplace_graph.add_node(node)
                    sameplace_pos[node] = pos[node]
                    if len(sameplace_graph.nodes) > len(sameplace_colormap):  # for if 3 at the same place
                        sameplace_colormap.append('tab:blue')

    return node_colormap, labels_at_step, sameplace_colormap, sameplace_graph, sameplace_pos


def plot_result(graph, pos, T, U, R, results_df, success, sensor_locations, labels):
    for route_to_vis in range(R):
        for step in range(T):

            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,5)) #, gridspec_kw={'height_ratios': [1, 0.6]})
            node_colormap, labels_at_step, sameplace_colormap, sameplace_graph, sameplace_pos = draw_nodes(graph, results_df, sensor_locations, U, R, step, labels, pos, route_to_vis)
            edge_colormap, edge_weightmap = draw_edges(graph, results_df, step, T, U, route_to_vis)

            nx.draw_networkx_nodes(G=graph, pos=pos,
                                   node_color=node_colormap, alpha=0.9, node_size=200,
                                   node_shape=matplotlib.markers.MarkerStyle(marker='o', fillstyle='full'),
                                   ax=ax[0])

            nx.draw_networkx_labels(G=graph, pos=pos, labels=labels_at_step, font_size=10, ax=ax[0])

            if len(sameplace_pos) > 0:
                nx.draw_networkx_nodes(G=sameplace_graph, pos=sameplace_pos,
                                       node_color=sameplace_colormap, alpha=0.9, node_size=200,
                                       node_shape=matplotlib.markers.MarkerStyle(marker='o', fillstyle='bottom'),
                                       ax=ax[0])

            nx.draw_networkx_edges(G=graph, pos=pos, edge_color=edge_colormap, width=edge_weightmap, ax=ax[0])

            labels_at_step.clear()
            #ax[0] = plt.gca()
            ax[0].set_aspect('equal')
            ax[0].set_facecolor('white')

            # legend

            legend_elements = [Line2D([0], [0], marker='o', color='w', label='pursuer',
                                      markerfacecolor='tab:blue', markersize=8),
                               Line2D([0], [0], marker='o', color='w', label='evader',
                                      markerfacecolor='tab:orange', markersize=8),
                               Line2D([0], [0], marker='o', color='w', label='sensor',
                                      markerfacecolor='#4caf50', markersize=8)]

            #fig.legend(handles=legend_elements, loc=7)
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 1.6))
            #ax[0].text(-0.9, 4, 't='+str(step), bbox=dict(facecolor='lightgray', alpha=0.5))
            #ax[0].text(-1.1, 3, 'route=' + str(route_to_vis), bbox=dict(facecolor='lightgray', alpha=0.5))
            #ax[0].text(-1.7, 2, 'intercepted=' + str(success[f'route{route_to_vis}']), bbox=dict(facecolor='lightgray', alpha=0.5))

            optimal_tree = plt.imread('figs/optimaltree.png')
            ax[1].imshow(optimal_tree)
            ax[1].axis('off')

            plt.savefig('figs/test_route'+str(route_to_vis) + '_step' + str(step) + '.png', bbox_inches='tight')
            plt.clf()

    with imageio.get_writer('figs/mygif.gif', mode='I', duration=0.5) as writer:
        for r in range(R):
            for filename in ['figs/test_route'+str(r)+'_step'+str(step)+'.png' for step in range(T)]:
                image = imageio.imread(filename)
                writer.append_data(image)

    # remove files
    for filename in set(['figs/test_route'+str(r)+'_step'+str(step)+'.png' for step in range(T) for r in range(R)]):
        os.remove(filename)
