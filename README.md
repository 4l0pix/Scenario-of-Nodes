# Scenario-of-Nodes
Every node owns a dataset with specific statistics like the mean, standard
deviation, clusters, etc
Datasets consist of multivariate vectors <x1, x2, x3, ..., xM>
New data arrive in every node
After the arrival of data, every node decides, apart from the local dataset, where
to send the data (to other nodes)
However, we do not want to flood the network and send the data to all nodes
(increases the burden of the network)
Decision: Send the data to ‘core peers’ after detecting the subgroups of the
network if the new data match to the remote dataset
For instance V11 may decide to send the data to V4, V1.
Reason:
A. We have to cluster the network to deliver the subgroups of nodes
B. In every subgroup, we have to detect the centrality of the participating
nodes and select one node to host the data
C. If we select the ‘central’ node in every sub-group, the remaining nodes in
the cluster may ask, if necessary the data, so we ‘push’ the data to other
parts of the network to enhance their datasets with positive impact on
their processing capabilities
Parts of the work

1. We create a graph of nodes and put data in every node (just select an
open dataset from Kaggle, uci, etc) – a couple of dimensions are ok – put
the number of nodes and dimensions into a global variable to have the
opportunity for multiple experiments
2. For every node, we get the statistics of data: we get 3 clusters over the
data and store their centroids and radius)
3. Adopt or create an algorithm for graph clustering – see here
https://en.wikipedia.org/wiki/Cluster_graph#:~:text=In%20graph
%20theory%2C%20a%20branch,called%20P3%2Dfree%20graphs.
4. Keep the subgroups somehow or annotate every node with the subgroup it
belongs to.
5. For each subgroup detect the most important nodes based on nodes
centrality

a. See here https://webs-deim.urv.cat/~sergio.gomez/papers/Gomez-
Centrality_in_networks-Finding_the_most_important_nodes.pdf

6. How to detect the important nodes: We calculate the Importance Index of
nodes as follows:
a. We calculate the degree centrality – it is affected by how many
edges (connections) nodes have
b. We calculate the Betweenness Centrality – a node which falls in the
communication paths between many pairs of nodes plays an
important role – it can support the distribution of data when
required
c. The two above metrics are affected by the location of every node so
we calculate the Katz Centrality
d. All the above values are normalized in [0,1] and we get the final
result through their mean

7. Now, we have the importance index for every node in every sub-group.
The plan is to send the data to the most important node. However, the
data can go there only if they match to the underlying dataset. So, we get
the distance between the new data and the clusters centroids as follows:
a. Get the Euclidean distance between the new data and every
centroid (3 in total)
b. So, you get three distances. You rely on the smallest one and
calculate the Data Distance Indicator, i.e., DDI = {0, if the distance
in greater than the radius of the cluster, 1/(1+exp(a.x-b) if the
distance is lower than the radius of the cluster, a,b are parameters
and x is the distance}. This means that if the distance is high, even
if a node is important in a topological perspective is not good to
host the data. The data can go to the next important node if the
distance with one of the clusters is very low (there is a trade off
here).
c. The DDI is multiplied by (we may apply a more complex scheme
later) the result of the importance

8. The final importance value after the multiplication between the result of
the Step 6 and the DDI is sorted in a descending order and we select the
first node in the list to host the new data
9. We place the data in the selected nodes (one for every subgroup) and
update the clusters for every affected node (not all – only those getting
data will update the available clusters)
