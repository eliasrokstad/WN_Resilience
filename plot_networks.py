import pandas as pd
import wntr
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns

def absoluteFilePaths(directory):
    for root, dirs, files in os.walk(os.path.abspath(directory)):
        return [os.path.join(root, file) for file in files]


def get_resilience(directory):

    results_files = absoluteFilePaths(directory)
    df = pd.DataFrame()
    for file in results_files:
        temp_df = pd.read_csv(file, index_col=0).groupby(['Dimension', 'Aspect', 'Threshold']).mean().reset_index()
        temp_df = temp_df[['Dimension', 'Aspect', 'Threshold', 'Reliability']].rename(columns={'Reliability': 'Resilience'})
        temp_df['Network'] = file.split('_')[1]
        df = df.append(temp_df, ignore_index=True)
    return df



resilience = get_resilience('./reliability')
#%%
resilience = resilience[resilience['Dimension'].isin(['Magnitude', 'Peak']) & resilience['Aspect'].isin(['Service', 'Spatial'])]
idxmin = resilience.groupby('Network')['Resilience'].sum().idxmin()
idxmax = resilience.groupby('Network')['Resilience'].sum().idxmax()
temp = resilience.loc[(resilience['Dimension'] == 'Magnitude') & (resilience['Aspect'] == 'Spatial') & (resilience['Threshold'] == 1), ['Resilience', 'Network']]
idxmax = temp.loc[temp['Resilience'].idxmax(), 'Network']
idxmin = temp.loc[temp['Resilience'].idxmin(), 'Network']

def get_graph(idx):
    wn = wntr.network.WaterNetworkModel(f'variants/variant_{idx}.inp')
    graph = wn.get_graph().to_undirected()

    pos = dict()
    for name in wn.node_name_list:
            node = wn.get_node(name)
            pos[name] = node.coordinates
    return graph, pos




sns.set_context("paper", font_scale=2)
plt.rcParams['figure.figsize'] = [15, 8]
sns.set_style("white")
sns.set_palette("bright")


fig, ax = plt.subplots(1, 2, dpi=100)

for i, j in enumerate([idxmin, idxmax]):
    # dereference index into valid data (needed here since some repeated rather
    # than creating more, to illustrate handling unknown amount of data)

    G, pos = get_graph(j)
    nx.draw_networkx_nodes(G, pos, ax=ax[i], node_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax[i])
    ax[i].set_title(f'Network variant {j}')


plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('variant_graphs.png', bbox_inches='tight')
plt.show()