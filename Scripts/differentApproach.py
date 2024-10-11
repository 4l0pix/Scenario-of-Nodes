# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import math
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")
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
TaskRanges = []


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
    data = data.sample(frac=1,random_state=42)
    data=data.head(2220)
    ogData = len(data)
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
   # print("Sample:",len(sample))
   # print("Centroids1:",len(centroids[0]))
   # print("Centroids2:",len(centroids[1]))
   # print("Centroids3:",len(centroids[2]))
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

def final_importance_degree(centers,radiuses,DDI,subgroup,final_i_d,importance_degree,new_data):
    for i in subgroup: # We get each nodes centroid and radius
        centroids = centers[i] 
        radius = radiuses[i]
        # For every sample we will need 3 distances
        for index,row in new_data.iloc[1:].iterrows():
            sample = row.values # We get each sample
            ddi,rad = euclid_dist(sample,centroids,radius)
            # The DDI list hosts [node_index, DDI, radius of the cluster]
            DDI.append([i,ddi,rad])
        subgroup = tuple(subgroup)       
        for j in range(len(subgroup)): # For each node in the subgroup (subgroup is a tuple of indexes)
            for k in DDI: # For every sublist in the DDI list
                if k[0] == subgroup[j]: # If the 1st column(node index) == the node we are currently dealing with    
                    final_i_d[j] = round((k[1]*importance_degree[k[0]]),4)           
    return final_i_d

            
def most_important_nodes(new_data):
    global MostImportantNodes, RandomMostImportantNodesData, RandomMostImportantNodes
    importance_degree = {node: 0 for node in range(num_nodes)}

    # Calculate and display the importance of nodes in each subgroup
    DDI = []
    MIN_set = {row[1] for row in MostImportantNodes}  # Convert to set for faster membership testing
    RMIN_set = {row[1] for row in RandomMostImportantNodes}  # Set for the random scenario

    for i in subgroups:
        i = list(i)  # Convert set to list for easier handling
        final_i_d = [0] * len(i)  # Initialize final importance degree list once per subgroup
        mean_dict = importance(graph, i)  # Calculate mean importance for each node in subgroup

        for idx, node in enumerate(i):
            # For the random scenario
            if node in most_important_nodes_rand and node not in RMIN_set:
                RandomMostImportantNodes.append([final_i_d[idx], node])
                RandomMostImportantNodesData.append(graph.nodes[node]['data'])
                RMIN_set.add(node)  # Update set to avoid future duplicate checks

            # Update importance degree for each node
            importance_degree[node] = mean_dict[node]

        # Calculate final importance degree
        final_i_d = final_importance_degree(centers, radiuses, DDI, i, final_i_d, importance_degree, new_data)
        
        # Find the node with the maximum importance degree
        max_fid = max(final_i_d)
        max_fid_index = final_i_d.index(max_fid)
        max_node = i[max_fid_index]

        # Append to MostImportantNodes if not already included
        if max_node not in MIN_set:
            MostImportantNodes.append([max_fid, max_node, i])
            MostImportantNodesData.append(graph.nodes[node]['data'])
            MIN_set.add(max_node)  # Update set for faster future checks

    # Dictionary to store the most important node per subgroup
    most_important = {}

    # Loop through each node and store the one with the highest degree per subgroup
    for degree, node_num, subgroup in MostImportantNodes:
        subgroup_key = tuple(subgroup)
        # Update if subgroup is new or degree is higher
        if subgroup_key not in most_important or degree > most_important[subgroup_key][0]:
            most_important[subgroup_key] = [degree, node_num, subgroup]

    # Convert to list of nodes with highest degree per subgroup
    MostImportantNodes = list(most_important.values())    


def new_data_importation(head,tail,data):
    return data.iloc[head:tail]

def update_nodes(new_data):
    head = 0
    tailSize = batchSize//len(MostImportantNodes)
    tail = tailSize
    for node in range(len(MostImportantNodes)):
        new_node_data = new_data.iloc[head:tail]
        #graph.nodes[maxSimIndx]['data'] = pd.concat([graph.nodes[maxSimIndx]['data'], new_node_data], ignore_index=False)
        NewIncomingData.append(new_node_data)
        #print("Data was added to node:" ,maxSimIndx, "with similarity:", maxSim)
        head = tail+1
        tail += tailSize
        
def remove_outliers(df):
    """
    Remove outliers from the DataFrame using the Interquartile Range (IQR) method.

    Parameters:
        df (pd.DataFrame): The input DataFrame from which to remove outliers.

    Returns:
        pd.DataFrame: A DataFrame without outliers.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_cleaned = df.copy()
    
    for col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 2 * IQR # The number here is the sensitivity/strictness of outlier kicking
        upper_bound = Q3 + 2 * IQR
        
        # Remove outliers
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned        

def calculate_widest_range(list1, list2):
    # Function to calculate total width of ranges in a list
    def total_range_width(ranges):
        total_width = 0
        for lower, upper in ranges:
            total_width += (upper - lower)  # Calculate width for each range
        return total_width

    # Calculate total widths for both lists
    total_width_list1 = total_range_width(list1)
    total_width_list2 = total_range_width(list2)
    
    # Avoid division by zero (if both widths are zero, return 0%)
    if total_width_list1 == 0 and total_width_list2 == 0:
        return "Both dataframes have equal and zero range."
    
    # If one of the lists has zero width, it's 100% narrower
    if total_width_list1 == 0:
        return "The dataframe that occured from the randomly seleted nodes is 100% wider than the dataframe that occured from DDI."
    if total_width_list2 == 0:
        return "The dataframe that occured from DDI is 100% wider than the dataframe that occured from the randomly seleted nodes."
    
    # Calculate the wider range and the percentage difference
    if total_width_list1 > total_width_list2:
        percentage_diff = ((total_width_list1 - total_width_list2) / total_width_list2) * 100
        print( f"The dataframe that occured from DDI is {percentage_diff:.2f}% wider than the dataframe that occured from the randomly seleted nodes.")
    else:
        percentage_diff = ((total_width_list2 - total_width_list1) / total_width_list1) * 100
        print( f"The dataframe that occured from the randomly seleted nodes is {percentage_diff:.2f}% wider than the dataframe that occured from DDI." )

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

def dataframe_cosine_similarity(df1, df2):
    """
    Computes the average cosine similarity between all row pairs in two DataFrames.

    Parameters:
    df1 (pd.DataFrame): First DataFrame, expected to have the same columns as df2.
    df2 (pd.DataFrame): Second DataFrame, expected to have the same columns as df1.

    Returns:
    float: The average cosine similarity between the two DataFrames.
    """
    # Check if both dataframes have the same columns
    if set(df1.columns) != set(df2.columns):
        raise ValueError("DataFrames must have the same columns to compute similarity")

    # Drop non-numeric columns (like 'diagnosis')
    df1_numeric = df1.select_dtypes(include=['number'])
    df2_numeric = df2.select_dtypes(include=['number'])
    
    # Compute cosine similarity matrix between all rows in df1 and df2
    similarity_matrix = cosine_similarity(df1_numeric, df2_numeric)
    
    # Compute average similarity by taking the mean of the similarity matrix
    average_similarity = similarity_matrix.mean()
    
    return average_similarity

# 3. Function to perform a statistical significance test on range widths
def statistical_significance_test(df1, df2):
    """
    Performs Levene's test to determine if there is a statistically significant
    difference in the range widths of the two dataframes.

    Parameters:
        df1 (pd.DataFrame): First dataframe.
        df2 (pd.DataFrame): Second dataframe.

    Returns:
        str: Result of Levene's test indicating significance level.
    """
    # Calculate ranges for each column
    df1_ranges = [col.max() - col.min() for _, col in df1.items()]
    df2_ranges = [col.max() - col.min() for _, col in df2.items()]
    # Perform Levene's test
    stat, p_value = stats.levene(df1_ranges, df2_ranges)
    if p_value < 0.05:
        return f"The range widths are significantly different (p-value = {p_value:.4f})."
    else:
        return f"The range widths are not significantly different (p-value = {p_value:.4f})."

    

# Initialize variables
total_manhattan_distance = 0.0
scaler = StandardScaler()

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
subgroups = nx.community.louvain_communities(graph, seed=np.random)
most_important_nodes_rand = [np.random.randint(0, num_nodes) for _ in range(len(subgroups))]

# Visualize the graph and subgroups
visualize_graph(graph)
visualize_subgroups(graph, subgroups)

# Clear unnecessary data from the previous iteration
del data
batchSize = 500
batchStart = 0  # This means we get a batch of values each time
batchStop = batchSize
MostImportantNodes = []  # A general-purpose list that contains most important nodes of each subgroup
MostImportantNodesData = []
RandomMostImportantNodes = []
RandomMostImportantNodesData = []


RestNodesData = []  # Ensure this is initialized each time
for node in range(num_nodes):
    if node in MostImportantNodes:
        MostImportantNodesData.append(StartingPointData[node])
    else:
        RestNodesData.append(StartingPointData[node])

data = pd.read_csv(path2)
start = time.time()

for j in range(5):  # Changed inner loop variable to avoid shadowing
    new_data = new_data_importation(batchStart, batchStop, data)
    most_important_nodes(new_data)
    update_nodes(new_data)
    batchStart = batchStop + 1
    batchStop += batchSize

# Sorting MostImportantNodes is redundant here since you only need to do this after all updates
MostImportantNodes.sort()

end = time.time()
del data


time_elapsed = end - start


MostImportantNodesData = pd.concat(MostImportantNodesData, axis=0, ignore_index=True)
# Outliers cleaning using IQR method
#MostImportantNodesData = remove_outliers(MostImportantNodesData)
RandomMostImportantNodesData = pd.concat(RandomMostImportantNodesData, axis=0, ignore_index=True)
RestNodesData = pd.concat(RestNodesData, axis=0, ignore_index=True)

# Normalize the data for accurate similarity calculation
Norm_MostImportantNodesData = normalize_dataframe(MostImportantNodesData)
Norm_RandomMostImportantNodesData = normalize_dataframe(RandomMostImportantNodesData)

# Calculate the Cosine Similarity of the two dataframes

similarity_score = dataframe_cosine_similarity(Norm_MostImportantNodesData, Norm_RandomMostImportantNodesData)

# Calculate ranges for most important nodes
MostImportantNodesRanges = [(MostImportantNodesData[col].min(), MostImportantNodesData[col].max()) for col in MostImportantNodesData.columns]
RandomMostImportantNodesRanges = [(RandomMostImportantNodesData[col].min(), RandomMostImportantNodesData[col].max()) for col in RandomMostImportantNodesData.columns]
# MUST ENTER THE DDI LIST FIRST AND THEN THE RANDOM LIST
# Call the function to compare the ranges and update counts
calculate_widest_range(MostImportantNodesRanges, RandomMostImportantNodesRanges)

# Print results for the single iteration
print("Cosine Similarity for this iteration:", similarity_score)
print(statistical_significance_test(MostImportantNodesData, RandomMostImportantNodesData))
