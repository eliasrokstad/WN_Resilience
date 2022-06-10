import wntr
import networkx as nx
import numpy as np
import pandas as pd
from os import path

def attributes(G):
    uG = G.to_undirected()
    sG = nx.Graph(uG)
    return [nx.density(G),
            nx.average_shortest_path_length(G),
            nx.transitivity(sG),
            wntr.metrics.central_point_dominance(G),
            nx.diameter(uG)/G.number_of_edges(),
            nx.radius(uG)/G.number_of_edges(),
            len(wntr.metrics.bridges(G))/G.number_of_edges()]



n=1
data = []
for i in range(1000):
    variant = f'variants/variant_{n}.inp'
    if path.exists(variant):
        wn = wntr.network.WaterNetworkModel(variant)
        array = [f'variant_{n}'] + attributes(wn.get_graph())
        data.append(array)
    else:
        print(variant + ' dont exists')

    n += 1

data = pd.DataFrame(data,
                    columns=['Variant',
                             'Link density',
                             'Average shortest path length',
                             'Clustering Coefficient',
                             'Central Point of Dominance',
                             'Graph diameter',
                             'Graph radius',
                             'Bridge density']
                    )
data.to_csv('attributes.csv', index=False)
