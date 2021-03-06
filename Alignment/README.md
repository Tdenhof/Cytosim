# Alignment

For the purpose of the simulation the linkage distance is the difference in two phi values of a actin filament. The method of hierarchical clustering is performed as follows.

At the start, treat each data point as one cluster. Therefore, the number of clusters at  the start will be K. (K is an integer representing the number of data points)

    Form a cluster by joining the two closest data points resulting in K-1 clusters.
    Form more clusters by joining the two closest clusters resulting in K-2 clusters.
    Repeat the above three steps until one big cluster is formed.
    
    
Then compute a dendrogram to find optimal number of clusters. Optimal number of clusters corresponds to the longest vertical distance without any horizontal line passing through. Once this distance is selected a horizontal line is drawn through it. The number of vertical lines this newly created horizontal line passes is equal to number of clusters. On an uniformly distributed actin mesh network the computed dendrogram returned 4 to be the optimal number of cluster. Therefore it was decided that 4 clusters would be made for each simulation. The third step was to filter out outliers based on position. To do this the package sklearn was used. From sklearn we used the sklearn.neighbors module with the LocalOutlierFactor (LOF) method. LOF is an unsupervised machine learning algorithm that uses the density of data points in the domain to detect outliers. LOF compares the density of any given data point to the density of its neighbors and the computed anomaly score of each point is called the Local Outlier Factor. It measures the local deviation of the density of a given point with respect to its neighbors. By comparing the local density of a point to the local densities of its neighbors, one can identify points that have a substantially lower density than their neighbors. These are considered outliers.

    - n is number of neighbors upto k total neighbors
    
    - LOF(X)=[(LRD(1st n) + LRD(2nd n) + ... + LRD(kth n))/LRD(X)]/k
    
    - LRD(X) = 1/(sum of Reachable Distance (X, n))/k) 
    
    - Reachable Distance = distance measured: we used euclidean method
    
The final step was to perform a cluster centroid distance analysis based off of the centroid of each of cluster. To compute the centroid of a cluster find the average position of all the points in each cluster. Thne use the distance between two points equation to find the distance from the centroid and every other point in the cluster. Note this is using posX and posY from the initial fiber position report report. Then turn this centroid distance data into a dataframe utilizing the DataFrame method from the pandas library. This data corresponds to the distance from every point in a cluster to the centroid of the cluster. Use this created dataframe to perform an Interquartile range analysis generating the  count, mean distance, Std, Min, 25 percentile,  50 percentile,  75 percentile, and max distance for each cluster. This is done using the describe() method from pandas which computes and displays summary statistics for a python dataframe. 
