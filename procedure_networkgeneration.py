import matplotlib.pyplot as plt
from scipy.spatial import distance
import copy
import networkx as nx
import numpy as np
import pandas as pd
import wntr
import pickle


class GenerationProcedure(object):
    """
    GenerationProcedure is a network generation procedure to produce network variants
    based on a benchmark model. The procedure is initialized with a input file (.inp)
    of a working water distribution network model.
    """
    def __init__(self, input):
        self.wn = wntr.network.WaterNetworkModel(input)
        self.roughness = list()
        self.links = list()
        for name in self.wn.pipe_name_list:
            link = self.wn.get_link(name)
            self.links.append(
                [name, (link.end_node.name, link.start_node.name),
                 link.diameter, link.roughness]
            )
        self.links = pd.DataFrame(self.links, columns=['ID', 'edge', 'd', 'r'])
        self.pos = dict()
        for name in self.wn.node_name_list:
            node = self.wn.get_node(name)
            self.pos[name] = node.coordinates

        self.remove_links = list()
        self.critical_nodes = self.wn.tank_name_list + self.wn.reservoir_name_list
        self.graph = self.wn.get_graph().to_undirected()
        self.service_areas(min_size=15)
        self.classification()

    def plot_graph(self, graph):
        nx.draw(graph, pos=self.pos, node_size=20)
        plt.show()

    def plot_graph_highlight(self, graph, nodes = [], links = [], node_size=20):
        node_color = []
        for name in nodes:
            if name in graph.nodes:
                node_color.append(name)
        link_color = []
        for name in links:
            if name in graph.edges:
                link_color.append(name)
        nx.draw(graph, pos=self.pos, node_size=node_size)
        nx.draw_networkx_nodes(graph, self.pos, nodelist=node_color, node_color='r', node_size=node_size)
        nx.draw_networkx_edges(graph, self.pos, edgelist=links, edge_color='r')
        plt.show()

    def service_areas(self, min_size=15):
        G = self.graph
        self.brigdes = self.wn.pump_name_list + self.wn.valve_name_list
        for name in self.wn.pipe_name_list:
            link = self.wn.get_link(name)
            if str(link.status) == 'Closed':
                self.brigdes.append(name)
            elif link._cv is True:
                self.brigdes.append(name)
            else:
                self.roughness.append(link.roughness)

        for brigde in self.brigdes:
            link = self.wn.get_link(brigde)
            G.remove_edge(link.start_node.name, link.end_node.name)
            if link.start_node.name not in self.critical_nodes:
                self.critical_nodes.append(link.start_node.name)
            if link.end_node.name not in self.critical_nodes:
                self.critical_nodes.append(link.end_node.name)

        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        self.dma = dict()
        for idx, g in enumerate(components, start=1):
            if len(g.nodes()) > min_size:
                dma = nx.Graph()
                dma.add_nodes_from(g.nodes(data=True))
                dma.add_edges_from(g.edges(data=True))
                self.dma[idx] = dma

    def classification(self):
        self.dma_critical_nodes = dict()
        self.dma_float_nodes = dict()
        self.dma_links = dict()
        self.backbone = dict()
        for dma_id in self.dma.keys():
            self.dma_float_nodes[dma_id] = list()
            self.dma_links[dma_id] = list()
            self.backbone[dma_id] = nx.Graph()
            self.backbone[dma_id].add_nodes_from(self.dma[dma_id].nodes())
            self.dma_critical_nodes[dma_id] = list()
            for n in self.dma[dma_id].nodes:
                if n in self.critical_nodes:
                    self.dma_critical_nodes[dma_id].append(n)

            weights = []
            for edge in self.dma[dma_id].edges():
                u, v = edge[0], edge[1]
                if (u,v) in self.links['edge'].to_list():
                    row = self.links[(self.links['edge'] == (u,v))]
                else:
                    row = self.links[(self.links['edge'] == (v,u))]
                self.dma_links[dma_id].append(
                    (u, v, self.links.loc[row.index[0], 'ID'])
                )
                weights.append((u, v, 1/self.links.loc[row.index[0], 'd']))

            self.dma[dma_id].add_weighted_edges_from(weights)
            for n1 in self.dma_critical_nodes[dma_id]:
                for n2 in self.dma_critical_nodes[dma_id]:
                    if n1 != n2:
                        path = nx.shortest_path(
                            self.dma[dma_id], source=n1, target=n2, weight='weight'
                        )
                        for i in range(len(path)-1):
                            self.backbone[dma_id].add_edge(path[i], path[i+1])

            for node in self.backbone[dma_id].nodes:
                if self.backbone[dma_id].degree[node] == 0:
                    self.dma_float_nodes[dma_id].append(node)

            for link in self.dma_links[dma_id]:
                if link[0] in self.dma_float_nodes[dma_id] \
                        or link[1] in self.dma_float_nodes[dma_id]:
                    self.remove_links.append(link[2])

    def generate_variant(self, dgr_exp=10.0, dst_exp=10.0, n_connections=10, alpha=45):
        temp_backbone = copy.deepcopy(self.backbone)
        temp_float_nodes = copy.deepcopy(self.dma_float_nodes)
        add_links = list()
        for dma_id in self.backbone.keys():
            dma = self.backbone[dma_id]
            float_nodes = self.dma_float_nodes[dma_id]
            connections = dict()
            for node_id in float_nodes:
                connections[node_id] = []
                for neighbour_id in dma.nodes():
                    if neighbour_id != node_id:
                        dst = distance.euclidean(
                            self.pos[node_id],
                            self.pos[neighbour_id]
                        )
                        connections[node_id].append([neighbour_id, dst])
                connections[node_id] = \
                    pd.DataFrame(connections[node_id], columns=['ID', 'dst'])
                connections[node_id] = \
                    connections[node_id].sort_values('dst').head(n_connections)
                connections[node_id]['dst'] = \
                    1/np.power(connections[node_id]['dst'], dst_exp)
                connections[node_id]['dst'] = \
                    connections[node_id]['dst']/connections[node_id]['dst'].sum()

            while nx.is_connected(dma) == False:
                float_pool = pd.DataFrame(
                    [[node, dma.degree[node]] for node in float_nodes],
                    columns=['ID', 'dgr']
                    )
                float_pool['dgr'] = 1/np.power((float_pool['dgr'] + 1), dgr_exp)
                float_pool['dgr'] = float_pool['dgr']/float_pool['dgr'].sum()
                u = float_pool.sample(1, weights='dgr')['ID'].tolist()[0]

                if len(connections[u]['ID']) == 0:
                    float_nodes.remove(u)
                    continue

                v_row = connections[u].sample(1, weights='dst')
                v_index = v_row.index.tolist()[0]
                connections[u] = connections[u].drop(v_row.index.tolist()[0])

                v = v_row['ID'].tolist()[0]
                include = True
                for v_ex in dma.neighbors(u):
                    v_1 = [
                        self.pos[u][0] - self.pos[v][0],
                        self.pos[u][1] - self.pos[v][1]
                    ]
                    v_2 = [
                        self.pos[u][0] - self.pos[v_ex][0],
                        self.pos[u][1] - self.pos[v_ex][1]
                    ]
                    radians = self._angle_between(v_1, v_2)
                    degrees = np.degrees(radians)
                    if abs(degrees) < alpha:
                        include = False
                        break

                if not include:
                    continue

                for v_ex in dma.neighbors(v):
                    v_1 = [
                        self.pos[v][0] - self.pos[u][0],
                        self.pos[v][1] - self.pos[u][1]
                    ]
                    v_2 = [
                        self.pos[v][0] - self.pos[v_ex][0],
                        self.pos[v][1] - self.pos[v_ex][1]
                    ]
                    radians = self._angle_between(v_1, v_2)
                    degrees = np.degrees(radians)
                    if abs(degrees) < alpha:
                        include = False
                        break

                if include:
                    add_links.append(([u, v]))
                    dma.add_edge(u, v)


        self.variant = copy.deepcopy(self.wn)
        for pipe in self.remove_links:
            self.variant.remove_link(pipe)

        self.pipes_added = list()
        roughness = round(np.array(self.roughness).mean(), 1)
        for n, con in enumerate(add_links):
            name = f'NP{n+1}'
            length = distance.euclidean(self.pos[con[0]], self.pos[con[1]])
            diameter = 0.08
            self.variant.add_pipe(
                name, start_node_name=con[0], end_node_name=con[1],
                length=length, diameter=diameter, roughness=roughness
            )
            self.pipes_added.append(name)

        self.backbone = temp_backbone
        self.dma_float_nodes = temp_float_nodes

    def calibrate(self, save=None):
        diameter = {0.08:0.1, 0.1:0.125, 0.125:0.15, 0.15:0.2, 0.2:0.25, 0.25:0.3, 0.3:0.4, 0.4:0.5, 0.5:0.6, 0.6:0.6}
        flow = {0.08: 1, 0.1:1, 0.125:1, 0.15:1.5, 0.2:1.5, 0.25:1.75, 0.3:1.75, 0.4:2, 0.5:2, 0.6:2}
        self.variant.options.hydraulic.demand_model = 'PDD'
        calibrating = True
        while calibrating:
            sim = wntr.sim.WNTRSimulator(self.variant)
            results = sim.run_sim()
            calibrating = False
            for name in self.pipes_added:
                link = self.variant.get_link(name)
                maximum = results.link['velocity'][name].max()
                d = link.diameter
                if maximum > flow[d]:
                    calibaring = True
                    link.diameter = diameter[d]
        if save != None:
            self.variant.write_inpfile(save, version=2.2)

    def _angle_between(self, v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


gen = GenerationProcedure('Ctown.inp')
txt_file = 'input_generation.txt'

for n in range(900):
    try:
        name = f'variant_{n+102}'
        print(name)
        dgr_exp = np.random.choice(np.arange(1, 10, 1))
        dst_exp = np.random.choice(np.arange(1, 10, 1))
        n_connections = np.random.choice(np.arange(5, 16, 1))
        alpha = np.random.choice(np.arange(0, 60, 1))

        gen.generate_variant(dgr_exp=dgr_exp, dst_exp=dst_exp, n_connections=n_connections, alpha=alpha)
        gen.calibrate(f'variants/{name}.inp')

        file = open(txt_file, "a")
        file.write(f'{name}, {dgr_exp}, {dst_exp}, {n_connections}, {alpha} \n')
        file.close()

    except (RuntimeError, TypeError, NameError, ValueError):
        gen = GenerationProcedure('Ctown.inp')
        continue