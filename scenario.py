# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import math
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the number of nodes in the graph
num_nodes = 15


# Function to create a random graph with nodes and edges
def graph_creation():
    # Create an empty graph
    graph = nx.Graph()
    
    # Generate random node positions with uniform distribution in a 2D space
    node_positions = np.random.uniform(-1, 1, size=(num_nodes, 2))
    
    # Add nodes to the graph with positions
    for i, pos in enumerate(node_positions):
        node_lbl = "N" + str(i)
        graph.add_node(i, pos=pos, label=node_lbl)
    
    # Add edges with a given probability of connection
    probability_of_connection = 0.25
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.uniform(0, 1) < probability_of_connection:
                graph.add_edge(i, j)
                
    return graph, node_positions

# Function to visualize the graph
def visualize_graph(graph):
    # Draw the graph using Matplotlib
    pos = {i: pos for i, pos in enumerate(node_positions)}
    nx.draw(graph, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=8, edge_color='black')
    plt.title('Random Graph with Uniform Distribution of Edges (-1/1)')
    plt.show()

# Function to match a dataset to the graph nodes
def match_dataset_to_graph(graph):
    for i in range(num_nodes):
        # Read the data from the "Cancer_Data.csv" file
        data = pd.read_csv("Cancer_Data.csv")
        # Map labels 'M' and 'B' to 0 and 1
        conversion_mapping = {'M': 0, 'B': 1}
        data["diagnosis"] = data["diagnosis"].map(conversion_mapping)
        data.drop(columns="id", axis=1, inplace=True)
        # Shuffle the data
        data = data.sample(frac=1, random_state=42)
        # Create subsets of the data for each graph node
        subset_size = len(data) // num_nodes
        for i in range(num_nodes):
            head = i * subset_size
            tail = (i + 1) * subset_size
            subset = data[head:tail]
            subset = subset.dropna(axis=1, how='any')
            # Store the subset in the graph node
            graph.nodes[i]['data'] = subset

# Function to perform K-Means clustering on data
def clustering(data):
    # Perform K-Means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(data)
    cluster_centers = kmeans.cluster_centers_
    # Calculate the radius for each cluster
    cluster_radii = []
    for i in range(3):
        cluster_points = data[cluster_labels == i]
        center = cluster_centers[i]
        radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
        cluster_radii.append(radius)
    cluster_labels = cluster_labels[:num_nodes]
   
    return cluster_centers, cluster_radii

# Function to calculate the importance of nodes in a subgroup
def importance(graph, subgroup):
    Subgraph = graph.subgraph(subgroup)
    # Calculate the Degree Centrality
    centrality_deg = nx.degree_centrality(Subgraph)
    # Calculate the Betweenness Centrality
    bet_centrality = nx.betweenness_centrality(Subgraph)
    # Calculate the Katz Centrality
    k_centrality = nx.katz_centrality(Subgraph)
    print("\nDegree Centrality-->", centrality_deg)
    print("Betweenness Centrality-->", bet_centrality)
    print("Katz Centrality-->", k_centrality)
    # Create a new dictionary for the means
    mean_dict = {}
    # Iterate through the keys
    for key in centrality_deg.keys():
        # Calculate the mean of the values for each key from the three dictionaries
        mean_value = np.mean([centrality_deg[key], bet_centrality[key], k_centrality[key]])
        # Store the mean value in the new dictionary
        mean_dict[key] = mean_value
    return mean_dict

# Function to visualize subgroups in the graph
def visualize_subgroups(graph, subgroups):
    subgroup_colors = {}

    # Assign colors to subgroups
    for i, subgroup in enumerate(subgroups):
        for node in subgroup:
            subgroup_colors[node] = i

    node_colors = [subgroup_colors.get(node, -1) for node in graph.nodes()]

    # Visualize the graph with nodes colored by subgroups
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, node_color=node_colors, with_labels=True, node_size=200, font_size=8, edge_color='black')
    plt.title(f'Subgroups\nNumber:{len(subgroups)}')
    plt.show()


def euclid_dist(sample,centroids):
    distances=[math.dist(sample,centroids[0]),math.dist(sample,centroids[1]),math.dist(sample,centroids[2])]
    print("\n")
    for index,i in enumerate (distances):
        print(f"Distance{index}--->",i)
    

    
###############################################################################    
    
    

# Create the graph and match the dataset to the nodes
graph, node_positions = graph_creation()
match_dataset_to_graph(graph)


######### Display the datasets associated with each graph node  ##########
#for i, node_id in enumerate(graph.nodes()):
    #print(graph.nodes[node_id]['label'], "--->", graph.nodes[node_id]['data'])

# Perform clustering for each node and display the results
centers=[]
for i, node in enumerate(graph.nodes()):
    data = graph.nodes[node]['data']
    cluster_centers, cluster_radii = clustering(data)
    centers.append(cluster_centers)
    #print("Node", graph.nodes[node]['label'], "centroids:", cluster_centers, "\nradius:", cluster_radii)

# Create subgroups using Louvain Modularity
subgroups = nx.community.louvain_communities(graph, seed=123)

# Visualize the graph and subgroups
visualize_graph(graph)
visualize_subgroups(graph, subgroups)

# Calculate and display the importance of nodes in each subgroup
for i in subgroups:
    mean_dict = importance(graph, i)
    print("Mean Importance Centrality Degrees--->",mean_dict)
    #max_key = max(mean_dict, key=mean_dict.get)
    #max_value = mean_dict[max_key]
    #print("The most important node is-->", max_key, " with overall centrality:", max_value)

new_data = pd.read_csv("Cancer_Data_New.csv")
# For every sample we will need 3 distances
for index,row in new_data.iloc[1:].iterrows():
    sample = row.values
    for centroid in centers:
        euclid_dist(sample,centroid)
