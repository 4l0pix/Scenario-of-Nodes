# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:41:25 2024

@author: athan
"""

# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import warnings
import time
import random  # Needed to control randomness in general

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Set a global random seed
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

#prvide the file paths
path1 = "Cancer_Data.csv"
path2 = "Cancer_Data_New_Big.csv"

###FOR METRICS###
num_nodes = 20          # Define the number of nodes in the graph
print(f"-----NODES{num_nodes}-----")
"""
When we change the number of nodes we also need to change the incoming data for training. 
Less nodes, less data.
In order to make a fair comparison. Also, when changing the num of training data 
we need to make an adjustment at the data batches.
"""
StartingPointData = []    # The data that are used as a starting point in the model
NewIncomingData = []     # All the new data that are used to update the most important node
MostImportantNodesData = [] # The data of the most important nodes that will help us determine the similarity score
RestNodesData = []

# Function to create a random graph with nodes and edges
def graph_creation():
    # Create an empty graph
    graph = nx.Graph()  
    # Generate random node positions with uniform distribution in a 2D space
    node_positions = np.random.uniform(-1, 1, size=(num_nodes, 2))    
    # Add nodes to the graph with positions
    for i, pos in enumerate(node_positions):
        graph.add_node(i, pos=pos, label="N" + str(i))  
    # Add edges with a given probability of connection (given 0.25 randomly)
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
    nx.draw(graph, pos, with_labels=True, node_size=200, node_color='grey', font_size=8, edge_color='black')
    plt.title('Random Graph with Uniform Distribution of Edges')
    plt.show()
    
# Function to match a dataset to the graph nodes
def match_dataset_to_graph(graph):
    global ogData
    # Read the data from the "Cancer_Data.csv" file
    data = pd.read_csv(path1)
    # Map labels 'M' and 'B' to 0 and 1
    conversion_mapping = {'M': 0, 'B': 1}
    data["diagnosis"] = data["diagnosis"].map(conversion_mapping)
    data.drop(columns="id", axis=1, inplace=True)
    # Shuffle the data
    data = data.sample(frac=1,random_state=SEED)
    #data=data.head(2220)
    #ogData = len(data)
    for i in range(num_nodes):
        # Create subsets of the data for each graph node
        subset_size = len(data) // num_nodes
        head = i * subset_size
        tail = (i + 1) * subset_size
        subset = data[head:tail]
        subset = subset.dropna(axis=1, how='any')
        # Store the subset in the graph node
        graph.nodes[i]['data'] = subset
        # We also store it in the StartingPointData list
        StartingPointData.append(graph.nodes[i]['data'])
        
# Function to perform K-Means clustering on data
def clustering(data):
    # Perform K-Means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=SEED)
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
    
    

def new_data_importation(head,tail,data):
    return data.iloc[head:tail]


def update_nodes(new_data):
    head = 0
    tailSize = batchSize//len(RandomMostImportantNodes)
    tail = tailSize    
    for node in RandomMostImportantNodes:
        new_node_data = new_data.iloc[head:tail]
        graph.nodes[node]['data'] = pd.concat([graph.nodes[node]['data'], new_node_data], ignore_index=False)
        NewIncomingData.append(new_node_data)
        #print("Data was added to node:" ,maxSimIndx, "with similarity:", maxSim)
        head = tail+1
        tail += tailSize
    
    
def normalize_dataframe(df):
    """
    Normalizes all numeric columns in a DataFrame using min-max scaling.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with normalized values between 0 and 1 for each numeric column.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Apply min-max normalization
    normalized_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
    
    # Retain non-numeric columns by merging them back into the normalized DataFrame
    for col in df.columns:
        if col not in numeric_df.columns:
            normalized_df[col] = df[col]
    
    return normalized_df

# Create the graph and match the dataset to the nodes
graph, node_positions = graph_creation()
match_dataset_to_graph(graph)

# Perform clustering for each node and display the results
centers = [] 
radiuses = []
for node in graph.nodes():
    data = graph.nodes[node]['data']
    cluster_centers, cluster_radii = clustering(data)
    centers.append(cluster_centers)
    radiuses.append(cluster_radii)

# Create subgroups using Louvain Modularity
subgroups = nx.community.louvain_communities(graph, seed=SEED)
most_important_nodes_rand = [np.random.randint(0, num_nodes) for _ in range(len(subgroups))]

# Visualize the graph and subgroups
visualize_graph(graph)
visualize_subgroups(graph, subgroups)


# Clear unnecessary data from the previous iteration
del data
batchSize = 500
batchStart = 0  # This means we get a batch of values each time
batchStop = batchSize


RandomMostImportantNodes = [random.randint(0, num_nodes-1) for _ in range(len(subgroups))]
RandomMostImportantNodesData = []


data = pd.read_csv(path2)
start = time.time()
for j in range(10):  # Changed inner loop variable to avoid shadowing
    new_data = new_data_importation(batchStart, batchStop, data)
    for i in subgroups:
        i = list(i)  # Convert set to list for easier handling
        for idx, node in enumerate(i):
            # For the random scenario
            if node in RandomMostImportantNodes:
                RandomMostImportantNodesData.append(graph.nodes[node]['data'])
    update_nodes(new_data)
    batchStart = batchStop + 1
    batchStop += batchSize
    
    
RandomMostImportantNodesData = pd.concat(RandomMostImportantNodesData, axis=0, ignore_index=True)
Norm_RandomMostImportantNodesData = normalize_dataframe(RandomMostImportantNodesData)
RandomMostImportantNodesRanges = [(RandomMostImportantNodesData[col].min(), RandomMostImportantNodesData[col].max()) for col in RandomMostImportantNodesData.columns]

# Stopping the watch
end = time.time()
time_elapsed = end - start
print("Time elapsed-->",time_elapsed)