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

## Anomaly detection

- Technique: Density estimation
    - Find region with high probability
    - Find region with low probability
    - Then you calculate prob. of the test data
        - And based on likeliness, decide.

- Example: Fraud detection
    - Often additional checks(manual) is performed after detection

### Gaussian (normal) distribution
- Probability of x is determied by a Gaussian with mean(μ) and standard deviation(σ)

- bell shaped curve
- Only works for just 1 number

### Anomaly detection algorithm (build upon Normal distribution)

- Consider n features:
    - probability of a feature vector is product of probability of each feature based on mean(μ) and standard deviation(σ)
        - This looks like this assumes independence but works for independent as well   
        - You need to calculate mean and standard deviation requried
- When a new test example come, you calculate probability and if it is less than or greater than (ε). (Which is reasonable probability)

### Developing and evaluating anomaly detection system

- Assume some labeled data of anomalous and non-anomalous examples: labeled y=0 and y=1
    - For all training examples, assume non-anomalous
    - Then cross validationa and test test with at least a few anomalous examples
    - Tune epsilon using cross validation set
    - Then use test set to test
> Often people use just train and cv sets

- Evaluation :
    - Fit model on training set
    - ON cv/test set, predict:
        - y = 0 or 1
    - This is often skewed (low anomalies)
    - Possible evaluation metrics:
        - Precision, recall, f-1 score
    
## Anomaly detection Vs supervised learning

- If you have some labelled test and cv sets, why not use supervised learning
- If very small number of labelled examples and there are different types of anomalies, anomaly detection might be better.
- If you dont have many types of anomaly (like spam detection), supervised learning might be better. Often if you have large enough data.

## Choosing what features to use

- More important in anomaly detection than in supervised learning
- Try to use more or less gausian features
    - If non-gaussian: 
        - You can reduce skewness using log transform.
        - Others also work log(x + c)

- Error analysis:
    - p(x) is comparable for normal and anomalous examples

- You can use vectorization to calculate mean, standard deviation, precision, recall, F1 score etc. 