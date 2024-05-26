import numpy as np
import pandas as pd
import networkx as nx
import math
from ts2vg import NaturalVG,HorizontalVG
import os

class featureExtractor:
    def __init__(self,data:dict) -> None:
        self.features = ("number of edges",
            "clustering coeff",
            "global efficiency",
            "graph index complexity",
            "max clique size",
            "TSP cost",
            "independence number",
            "min cut size",
            "vertex coloring number",
            "entropy",
            "avg degree",
            "density",
            "degree centrality",
            "betweenness centrality",
            "closeness centrality",
            "pagerank",
            "degree distribution",
            "average path length",
            "assortativity",
            "longest path")
        
        self.channels = ("I", "II", "III", "aVL", 
                        "aVR", "aVF", "V1", "V2", 
                        "V3", "V4", "V5", "V6")
        self.data = data
        self.visibilityG = NaturalVG()
        self.horizontalVG =HorizontalVG()
        
    def _GraphIndexComplexity(self,G):
        lamda_max = max(np.linalg.eig(nx.to_numpy_array(G))[0])
        n = G.number_of_nodes()

        c = (lamda_max - 2 * np.cos(np.pi/(n + 1))) / \
            (n - 1 - np.cos(np.pi/(n + 1)))
        C = 4 * c * (1 - c)

        return C


    def _VertexColoringNumber(self,G):
        greedy_color = nx.coloring.greedy_color(G)
        return len(set(greedy_color.values()))


    def _degree_distribution(self,G):
        vk = dict(G.degree())
        vk = list(vk.values())  # We get only the degree values
        maxk = np.max(vk)
        kvalues = np.arange(0, maxk + 1)  # Possible values of k
        Pk = np.zeros(maxk + 1)  # P(k)
        for k in vk:
            Pk[k] += 1
        Pk = Pk / sum(Pk)  # The sum of the elements of P(k) must be equal to one

        return kvalues, Pk


    def _shannon_entropy(self,G):
        k, Pk = self._degree_distribution(G)
        H = 0
        for p in Pk:
            if p > 0:
                H -= p * math.log(p, 2)
        return H

    def _features(self,G):
        no_of_edges = nx.number_of_edges(G)

        clustering_coeff = nx.average_clustering(G)

        global_efficiency = nx.global_efficiency(G)

        # small_worldness = nx.sigma(G)  # small worldness

        graph_index_complexity = self._GraphIndexComplexity(G)

        max_clique_size = len(next(nx.find_cliques(G)))  # size of max clique

        # cost of TSP
        tsp_cost = len(nx.approximation.traveling_salesman_problem(G))

        independence_number = len(
            nx.maximal_independent_set(G))  # independence number

        min_cut_size = len(nx.minimum_edge_cut(G))  # size of minimum cut

        vertex_coloring_number = self._VertexColoringNumber(G)

        entropy = self._shannon_entropy(G)
        
        avg_degree = np.mean([d for n, d in G.degree()])
        density = nx.density(G)
        clustering_coeff = nx.average_clustering(G)
        # Centrality measures
        degree_centrality = np.mean(list(nx.degree_centrality(G).values()))
        betweenness_centrality = np.mean(list(nx.betweenness_centrality(G).values()))
        closeness_centrality = np.mean(list(nx.closeness_centrality(G).values()))
        pagerank = np.mean(list(nx.pagerank(G).values()))
        # Other G properties
        degree_distribution = np.mean(np.array(nx.degree_histogram(G)))
        average_path_length = nx.average_shortest_path_length(G)
        assortativity = nx.degree_assortativity_coefficient(G)
        longest_path = nx.diameter(G)
        
        return (
        no_of_edges,
        clustering_coeff,
        global_efficiency,
        graph_index_complexity,
        max_clique_size,
        tsp_cost,
        independence_number,
        min_cut_size,
        vertex_coloring_number,
        entropy,
        avg_degree,
        density,
        degree_centrality,
        betweenness_centrality,
        closeness_centrality,
        pagerank,
        degree_distribution,
        average_path_length,
        assortativity,
        longest_path
    )
        
    def get_VG_feature_map(self):
        val = {}
        for i, j in self.data.items():
            sz = j.shape
            feature = np.zeros((sz[0], 12, len(self.features)))
            for k in range(sz[0]):
                for l in range(12):
                    self.visibilityG.build(j[k][l])
                    graph = self.visibilityG.as_networkx()
                    fe = self._features(graph)
                    feature[k][l] = fe
                    print(f'in - {i}-{k}-{l}')
            val[i] = feature
        return val
    def get_HVG_feature_map(self):
        val = {}
        for i, j in self.data.items():
            sz = j.shape
            feature = np.zeros((sz[0], 12, len(self.features)))
            for k in range(sz[0]):
                for l in range(12):
                    self.horizontalVG.build(j[k][l])
                    graph = self.horizontalVG.as_networkx()
                    fe = self._features(graph)
                    feature[k][l] = fe
                    print(f'in - {i}-{k}-{l}')
            val[i] = feature
        return val
    def save_map(self,dic:dict,root:str,name="g"):
        if not os.path.exists(root):
            os.mkdir(root)
        for i,j in dic.items():
            pth = os.path.join(root,i)
            if not os.path.exists(pth):
                os.mkdir(pth)
            np.save(os.path.join(j,name),j)
    
    def load_as_dataframe(self,dic:dict):
        data_rows = []

        # Iterate through each class and their data
        for class_label, dataset in dic.items():
            num_samples = dataset.shape[0]

            for sample_index in range(num_samples):
                row = {}
                row['Class'] = class_label

                for channel_index in range(12):
                    for feature_index in range(9):
                        column_name = f'{self.channels[channel_index]}_{self.features[feature_index]}'
                        row[column_name] = dataset[sample_index, channel_index, feature_index]

                data_rows.append(row)
        return pd.DataFrame(data_rows)
        
    def generate_praquet(self,df,path):
        df.to_parquet(path, index=False)

        
        
    def load_praquet(self,path:str):
        return pd.read_parquet(path)