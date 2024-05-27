# Estimation and Inference

- Estimation: Using sample data to approximate the value of an unknown population parameter.
- Inference: Process of drawing conclusions or making predictions about a population based on sample data. It also involves putting an accuracy on the estimate.

## Machine learning and statistical inference

- Similar, in the sense that we are using data to learn/infer qualitiese of a distribution that generated the data.
- We may care either about the whole distribution or just features.

## Parametric Vs Non-Parametric

- A parametric model is type of statistical model is a set of distributions or regressions, but have finite number of parameters and have strict assumptions about the distribution. For eg: normal distribution.

- Non-parametric model will make fewer assumptions. Eg. Creating distribution using histogram (cumulative distribution function).
    - We are not assuming normal or in fact any other distribution.

## Maximum likelihood estimation.

- Parametric model that estimates parameters based on likelihood function that is related to probability and is function of the parameters of the model.

## Common distributions:

- Uniform distribution
    - Equal chance that you will get a value in a range.

- Normal/Gaussian distribution 
    - Values closer to mean will have higher chance of occuring.

- Central limit theorem
    - Distribution of averages of sample is also normal distribution.

- Log normal distribution
    - The distribution becomes normal if we take log of the variable.
    - Larger outliers => larger tails
    - Small SD => closer to normal

- Exponential distributions
    - probability changes exponentially (decreasing or increasing)

- Poissons distribution
    - Number of occurence in a particular interval of time

## Bayesian vs Frequentist statistics

- A frequentist is converned with repeated observations in the limit.
    - Processes may have true frequencies, but we are interested in modeling probabilities as many, many repeats of an experiment.
    - Steps:
        - Derive the probabilistic property of a procedure
        - more data => more confidence
        - Apply the probability directly to the observed data
    
- A Bayesian describes parameters by probability distributions.
    - Before seeing data, a prior distribution(belief based) is formulated.
    - Updated after seeing data.
    - After updating, we get posterior distribution.

> Main element that differs is the interpretation, math is almost same.