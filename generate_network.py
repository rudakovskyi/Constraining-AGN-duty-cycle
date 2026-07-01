import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import pandas as pd
import igraph as ig


def generate_network(data_pos, linkinglength):
    data_pos = np.asarray(data_pos, dtype=float)
    if data_pos.ndim != 2 or data_pos.shape[1] != 3:
        raise ValueError(f"data_pos must have shape (N, 3). Got {data_pos.shape}.")
    if linkinglength <= 0:
        raise ValueError("linkinglength must be > 0.")

    N = data_pos.shape[0]
    G = nx.Graph()

    # Add nodes with coordinates
    for i in range(N):
        x, y, z = data_pos[i]
        G.add_node(i, pos=(x, y, z), posx=float(x), posy=float(y), posz=float(z))

    # KDTree for fast neighbor search
    point_tree = KDTree(data_pos)

    # Build edges
    for i in range(N):
        neighbors = point_tree.query_ball_point(data_pos[i], linkinglength)
        for j in neighbors:
            if i < j:
                d = float(np.linalg.norm(data_pos[i] - data_pos[j]))
                G.add_edge(i, j, length=d, weight=d)

    return G


def compute_node_metrics(G):
    """
    Compute local graph metrics and return them as a pandas DataFrame.

    Parameters
    ----------
    G : networkx.Graph

    Returns
    -------
    pandas.DataFrame
        Node-level metrics.
    """

    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    nodes = list(G.nodes())

    # ---------- Metrics ----------
    av_neighbor_deg = nx.average_neighbor_degree(G)

    # --- igraph conversion for betweenness (faster) ---
    g = ig.Graph.from_networkx(G)

    if G.number_of_edges() > 0 and "length" in next(iter(G.edges(data=True)))[2]:
        g.es["weight"] = [d["length"] for _, _, d in G.edges(data=True)]
        betweenness_vals = g.betweenness(weights="weight")
    else:
        betweenness_vals = g.betweenness()

    betweenness = dict(zip(nodes, betweenness_vals))

    # networkx metrics
    closeness = nx.closeness_centrality(G, distance="length")
    harmonic = nx.harmonic_centrality(G, distance="length")
    clustering = nx.clustering(G)   # uses topology (safer for distance graphs)

    # ---------- Build DataFrame ----------
    df = pd.DataFrame({
        "node": nodes,
        "degree": [G.degree(n) for n in nodes],
        "average_neighbor_degree": [av_neighbor_deg[n] for n in nodes],
        "closeness": [closeness[n] for n in nodes],
        "harmonic": [harmonic[n] for n in nodes],
        "betweenness": [betweenness[n] for n in nodes],
        "clustering": [clustering[n] for n in nodes],
    })

    return df

def centrality_cdf(df, centrality):
    
    
    '''An auxiliary function to compute centrality cdfs from dataframes'''
    
    if centrality not in df.columns:
        raise KeyError(f"Column {centrality} not found in DataFrame")
    column = centrality
    bins = 50
    if column == 'degree':
        
        bins = np.arange(0,40)
        
    if column == 'average_neighbor_degree':
        bins = np.arange(0,40)

    if column == 'betweeness':
        
        bins=np.append([0], np.logspace(-1,4,51))
        
    if column == 'harmonic':
        
        bins = np.arange(0,200)
        
    if column == 'closeness':
        
        bins=np.linspace(0,1e-1,41)

    if column == 'clustering':
        
        bins = np.arange(0,1.,0.05)
        
    hist_base, bin_edges = np.histogram(df[column].to_numpy(), bins=bins,density=True)
    hist_int = np.cumsum(hist_base * np.diff(bin_edges))
    
    return hist_int, bin_edges

