import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import wntr
import networkx as nx
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

data = pd.read_csv('attributes.csv')
data = data.head(300)
data['Graph diameter'] = data['Graph diameter']/388
data['Graph radius'] = data['Graph radius']/388

benchmark = 'Ctown.inp'
wn = wntr.network.WaterNetworkModel(benchmark)
benchmark = attributes(wn.get_graph())

columns = list(data.columns)[1:]
df = pd.DataFrame()

sns.set_context("paper", font_scale=1.8)
plt.rcParams['figure.figsize'] = [15, 5]
sns.set_style("white")
sns.set_palette("bright")
rename = {'Link density': 'd',
           'Average shortest path length': 'a',
            'Graph diameter': r'$C_c$',
           'Clustering Coefficient  ': r'$C_d$',
           'Graph radius': r'$G_d$',
           'Central Point of Dominance': r'$G_r$',
           'Bridge density': r'$B_d$'}
df = df.rename(columns=columns)

for i, att in enumerate(columns):
    temp = pd.DataFrame()
    temp['Value'] = (data[att]-data[att].min())/(data[att].max() - data[att].min())
    temp['Attribute'] = rename[att]
    df = df.append(temp)

g = sns.boxplot(x=df['Attribute'], y=df['Value'])
g.set_yticks([])
g.set_xlabel('Topological attribute')
plt.subplots_adjust(wspace=0.5)
plt.savefig('dist_att.png', bbox_inches='tight')
plt.close()



#%%
g = sns.FacetGrid(df, row="Attribute", legend_out=True, sharex=False, sharey=False, size=2, aspect=5)
g.map(sns.histplot, 'Value', bins=10)

for a in range(len(g.axes)):
    g.axes[a, 0].set_title('')
    g.axes[a, 0].set_xlabel(columns[a])
    g.axes[a, 0].axvline(x=benchmark[a], color='r', label='Benchmark model')
    if a == 0:
        g.axes[a, 0].legend()

plt.show()






