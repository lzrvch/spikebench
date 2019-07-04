import numpy as np
import pandas as pd


def abs_energy(x):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return sum(x * x)


def count_above_mean(x):
    """
    Returns the number of values in x that are higher than the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    x = np.asarray(x)
    m = np.mean(x)
    return np.where(x > m)[0].shape[0]


def count_below_mean(x):
    """
    Returns the number of values in x that are lower than the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    x = np.asarray(x)
    m = np.mean(x)
    return np.where(x < m)[0].shape[0]


def kurtosis(x):
    """
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.kurtosis(x)


def large_standard_deviation(x, r):
    """
    Boolean variable denoting if the standard dev of x is higher
    than 'r' times the range = difference between max and min of x.
    Hence it checks if

    .. math::

        std(x) > r * (max(X)-min(X))

    According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    x = np.asarray(x)
    return np.std(x) > (r * (max(x) - min(x)))


def maximum(x):
    """
    Calculates the highest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return max(x)


def minimum(x):
    """
    Calculates the lowest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return min(x)


def median(x):
    """
    Returns the median of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.median(x)


def mean(x):
    """
    Returns the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.mean(x)


def quantile(x, q):
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param q: the quantile to calculate
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.quantile(x, q)


def ratio_beyond_r_sigma(x, r):
    """
    Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

    :param x: the time series to calculate the feature of
    :type x: iterable
    :return: the value of this feature
    :return type: bool
    """
    x = np.asarray(x)
    return sum(abs(x - np.mean(x)) > r * np.std(x)) / len(x)


def ratio_value_number_to_time_series_length(x):
    """
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns

        # unique values / # values

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    if len(x) == 0:
        return np.nan

    return len(set(x)) / len(x)


def skewness(x):
    """
    Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1).

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.skew(x)


def standard_deviation(x):
    """
    Returns the standard deviation of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.std(x)


def sum_values(x):
    """
    Calculates the sum over the time series values

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return np.sum(x)


def symmetry_looking(x, param):
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    x = np.asarray(x)
    mean_median_difference = abs(np.mean(x) - np.median(x))
    max_min_difference = max(x) - min(x)
    return mean_median_difference < (param * max_min_difference)


def variance_larger_than_standard_deviation(x):
    """
    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return np.var(x) > np.std(x)


def variance(x):
    """
    Returns the variance of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.var(x)


def length(x):
    """
    Returns the length of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: int
    """
    return len(x)


def distribution_feature_vector(x):

    features = np.zeros(shape=44)

    feature_calculators = [
        abs_energy,
        count_above_mean,
        count_below_mean,
        kurtosis,
        length,
        maximum,
        minimum,
        mean,
        median,
        skewness,
        standard_deviation,
        sum_values,
        variance,
        variance_larger_than_standard_deviation,
        lambda x: large_standard_deviation(x, r=0.05)
    ]

    for q in np.linspace(0.1, 0.9, 9):
        feature_calculators.append(lambda x: quantile(x, q=q))

    for r in [0.5, 1., 1.5, 2.]:
        feature_calculators.append(lambda x: ratio_beyond_r_sigma(x, r=r))

    for r in np.linspace(0.2, 0.95, 16):
        feature_calculators.append(lambda x: symmetry_looking(x, param=r))

    for index, calculator in enumerate(feature_calculators):
        features[index] = calculator(x)

    return features
