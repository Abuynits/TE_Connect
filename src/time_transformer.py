# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# next steps: implement a time transformer for this model

"""
Popular time series preprocessing techniques include:
Just scaling to [0, 1] or [-1, 1]
Standard Scaling (removing mean, dividing by standard deviation)
Power Transforming (using a power function to push the data to a more normal distribution, typically used on skewed data / where outliers are present)
Outlier Removal
Pairwise Diffing or Calculating Percentage Differences
Seasonal Decomposition (trying to make the time series stationary)
Engineering More Features (automated feature extractors, bucketing to percentiles, etc)
Resampling in the time dimension
Resampling in a feature dimension (instead of using the time interval, use a predicate on a feature to re-arrange your time steps — for example when recorded quantity exceeds N units)
Rolling Values
Aggregations
Combinations of these techniques

"""