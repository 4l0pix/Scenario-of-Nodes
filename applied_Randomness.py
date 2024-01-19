# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import math
import time
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import statistics
# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")
path1 = "Cancer_Data.csv"
path2 = "Cancer_Data_New.csv"
###FOR METRICS###
num_nodes = 20             # Define the number of nodes in the graph
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
    nx.draw(graph, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=8, edge_color='black')
    plt.title('Random Graph with Uniform Distribution of Edges (-1/1)')
    plt.show()

# Function to match a dataset to the graph nodes
def match_dataset_to_graph(graph):
    # Read the data from the "Cancer_Data.csv" file
    data = pd.read_csv(path1)
    # Map labels 'M' and 'B' to 0 and 1
    conversion_mapping = {'M': 0, 'B': 1}
    data["diagnosis"] = data["diagnosis"].map(conversion_mapping)
    data.drop(columns="id", axis=1, inplace=True)
    # Shuffle the data
    data = data.sample(frac=1, random_state=42)
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
    #print("\nDegree Centrality-->", centrality_deg)
    #print("Betweenness Centrality-->", bet_centrality)
    #print("Katz Centrality-->", k_centrality)
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

def euclid_dist(sample,centroids,radiuses):
    b = -5
    a = b/2   # Parameters for the DDI
    
    # We calculate the euclidean distances between each sample and each node centroid
    distances=[math.dist(sample,centroids[0]),math.dist(sample,centroids[1]),math.dist(sample,centroids[2])]
    #print("\n")
    #for index,i in enumerate (distances):
        #print(f"Distance{index}--->",i)
    closest_dist = min(distances) # We get the closest distance
    #furthest_dist = max(distances)  # And the greatest distance
    rad_index = distances.index(closest_dist) # And the radius of the cluster with the closest distance
    ddi = 1/1+math.exp((a*closest_dist)-b) if (closest_dist <= radiuses[rad_index]) else 0 
    #print(f'[{closest_dist,furthest_dist}]')
    #print("X-->",closest_dist,"DDI-->",ddi)
    return ddi, radiuses[rad_index]

def final_importance_degree(centers,radiuses,DDI,i,final_i_d,importance_degree,new_data):
    for ii in i: # We get each nodes centroid and radius
        centroids = centers[ii] 
        radius = radiuses[ii]
        # For every sample we will need 3 distances
        for index,row in new_data.iloc[1:].iterrows():
            sample = row.values # We get each sample
            ddi,rad = euclid_dist(sample,centroids,radius)
            # The DDI list hosts [node_index, DDI, radius of the cluster]
            DDI.append([ii,ddi,rad])
        i = tuple(i)       
        for j in range(len(i)): # For each node in the subgroup (i is a tuple of indexes)
            for k in DDI: # For every sublist in the DDI list
                if k[0] == i[j]: # If the 1st column(node index) == the node we are currently dealing with    
                    final_i_d[j] = round((k[1]*importance_degree[k[0]]),4)           
    return final_i_d




            
def most_important_nodes(new_data):
    global MostImportantNodes, MostImportantNodesData, RandomMostImportantNodes, SmartMostImportantNodes
    importance_degree = {node:0 for node in range(15)}
    # Calculate and display the importance of nodes in each subgroup
    #print(new_data)
    DDI = []
    for i in subgroups:
        #print("\nSubgroup--->",i)
        i = list(i) # We turn the set into a list to be easier to handle
        final_i_d = [0]*len(i)   # Final importance degree
        mean_dict = importance(graph, i)
        # For every node we calculate the mean importance degree
        for node in range(num_nodes):
            if node in i:    
                # We save it into a dictionary in order to use it later in the DDI
                importance_degree[node] = mean_dict[node]
        final_i_d = final_importance_degree(centers,radiuses,DDI,i,final_i_d,importance_degree,new_data)
        #print("Importance degrees--->",final_i_d)
        max_fid = max(final_i_d)
        #print("Max--->",max_fid)
        max_fid_index = final_i_d.index(max_fid)
        SmartMostImportantNodes.append([max_fid_index,max_fid])
        for node in i:
            if node in most_important_nodes_rand:
                RandomMostImportantNodes.append([node,final_i_d[i.index(node)]])
        
        MostImportantNodes.append(i[max_fid_index]) if i[max_fid_index] not in MostImportantNodes else MostImportantNodes


def new_data_importation(head,tail,data):
    return data.iloc[head:tail]

def update_nodes(new_data,OverallSimilarity):
    head = 0
    tail = 1250//len(MostImportantNodes)
    for node in MostImportantNodes:
        new_node_data = new_data.iloc[head:tail]
        maxSim,maxSimIndx = sim4importN(new_node_data)
        graph.nodes[maxSimIndx]['data'] = pd.concat([graph.nodes[maxSimIndx]['data'], new_node_data], ignore_index=False)
        NewIncomingData.append(new_node_data)
        #print("Data was added to node:" ,maxSimIndx, "with similarity:", maxSim)
        OverallSimilarity.append(maxSim)
        head = tail + 1
        tail += tail
        
        
        
def sim4importN(new_node_data):  # Similarity for important Nodes
    #print("\n\nNew Batch of data")
    maxSim = 0  
    maxSimIndx = -1 # If -1 is returned then we have an error
    for i in MostImportantNodes:
            #print(cosine_similarity(new_node_data,graph.nodes[i]['data']))
            if cosine_similarity(new_node_data,graph.nodes[i]['data']).mean() > maxSim:
                maxSim = cosine_similarity(new_node_data,graph.nodes[i]['data']).mean()
                maxSimIndx = i
    return maxSim, maxSimIndx
            
###############################################################################    
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
    #print("Node", graph.nodes[node]['label'], "centroids:", cluster_centers, "\nradius:", cluster_radii)
# Create subgroups using Louvain Modularity
subgroups = nx.community.louvain_communities(graph, seed=np.random)
most_important_nodes_rand = [np.random.randint(0, num_nodes) for _ in range(len(subgroups))]
# Visualize the graph and subgroups
visualize_graph(graph)
visualize_subgroups(graph, subgroups)
del(data)
batchStart = 0 # This means we get the a batch of 500 values each time
batchStop = 2500
MostImportantNodes = [] # A general purpose list that contains most important nodes of each subgroup
RandomMostImportantNodes = []
SmartMostImportantNodes = []
OverallSimilarity=[]
data = pd.read_csv(path2)
start = time.time()

#for i in range():
    #print("Round",i)
new_data = new_data_importation(batchStart,batchStop,data)
most_important_nodes(new_data)
update_nodes(new_data,OverallSimilarity)
batchStart = batchStop + 1
batchStop += 2500
end = time.time()
del(data)
for node in range(num_nodes):
    if node in MostImportantNodes:
        MostImportantNodesData.append(StartingPointData[node])
    else:
        RestNodesData.append(StartingPointData[node])
time_elapsed = end-start
NewIncomingData = pd.concat(NewIncomingData, axis=0, ignore_index=True)
MostImportantNodesData = pd.concat(MostImportantNodesData, axis=0, ignore_index=True)
RestNodesData = pd.concat(RestNodesData, axis=0, ignore_index=True)
#similarity_matrix1 = cosine_similarity(MostImportantNodesData, NewIncomingData)
#similarity_matrix2 = cosine_similarity(RestNodesData, NewIncomingData)
print("Most Important Nodes Values(combined)", len(MostImportantNodesData))
print("New Incoming Values---> ", len(NewIncomingData))
#print("Cosine Similarity Score (betweeen the most import. nodes and new data---> ", "{:.4f}".format(similarity_matrix1.mean()))
#print("Cosine Similarity Score (betweeen the rest of the nodes and new data---> ", "{:.4f}".format(similarity_matrix2.mean()))

OverallSimilarity=statistics.mean(OverallSimilarity)
print("OverallSimilarity--->",OverallSimilarity)
print("\nSMART MODEL IMPORTANCE DEGREES")
for i in range(len(subgroups)):
    print("Smart model:The most important node--->",SmartMostImportantNodes[i][0], "with degree:",SmartMostImportantNodes[i][1])

print("\nRANDOM MODEL IMPORTANCE DEGREES")
for i in range(len(subgroups)):
    print("Random model:The most important node--->",RandomMostImportantNodes[i][0], "with degree:",RandomMostImportantNodes[i][1])

print("\nTime elapsed---> ", time_elapsed)
