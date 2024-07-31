# Unsupervised learning

> Working with unlabelled dataset.

## Beyond Supervised learning:

- Unsupervised Learning:
    - Clustering
    - Anamaly detection
- Recommendation Systems
- Reinforcement Learning

# Clustering

> Grouping inputs based on similarity

- Applications:
    - Market segmentation
    - Grouping similar news
    - DNA analysis
    - Astronomical data analysis

## K-means clustering

- Steps:
    - Decide on two random points i.e. cluster centroids (if you want two clusters)
    - Repeat:
        - For each point, check if the point is closer to cluster centroid 1 or cluster centroid 2.
        - The for each points closer to centroid 1, take average of the value and find new centroid. Do same for other centroids.

- Randomly initializing K cluster centroids
- Repeat {
    - Assign points to cluster centroids
        - for i = 1 to m:
            - c(i) := index (from 1 to K) of cluster closest to x(i)
    - Move cluster centroids:
        - for k = 1 to K
            m(k) := average(mean) of points assigned to cluster k
}

- Optimization objective:

$$ 
J = 1/m \sum_{i=1}^{m} \mathbb \| x^{(i)} - \mu_c{(i)} \|^2
$$

- where, mu_c is the cluster centroid to which example x(i) has been assigned.

- How to initialize:
    - choose K < m (no. of clusters < no. of training example)
    - Randomly pick K training examples and get their mean and use that as cluster centroid.
> This might result in getting stuck at local minima.

- Another way:
    - Run with different initializattion and use the option with smallest cost function
    - For i = 1 to 100 {
        - Randomly initialize k-mmeans
        - Run k means
        - Compute cost functions
    }
    - Pick set of clusters that gave lowest cost J

- Choosing the number of clusters:
    - You don't have the unambiguous answer.
    - Methods:
        - Elbow method:
            - Run with various values of k and chose the value where the rate of decrease of cost function is low.
            - This is ambiguous a lot of time.
        - Evaluate k-means based on how well it performs on the later purpose.