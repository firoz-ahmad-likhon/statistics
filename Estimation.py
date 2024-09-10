import scipy.stats as stats
import math


def validate_estimation_inputs(p, n):
    """
    Check the condition if it meets the constraints
    :param p: Proportion
    :param n: Sample size
    :return: Tuple of boolean and string message
    """
    success = n * p
    fail = n * (1 - p)
    if success < 10 or fail < 10:
        return False, "For proportion estimation, success and fail must both be at least 10."

    return True, "Inputs are valid."


def z_distribution(mean, SE, cl, range_start, range_end):
    """
    Estimate using Z Distribution.
    :param mean: Sample Mean (float)
    :param SE: Standard Error (float)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param range_start: Start of the range for probability calculation (float or None)
    :param range_end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the value falling within the specified range (float or None)
    """
    alpha = (1 - cl) / 2  # Significance level
    z_critical = stats.norm.ppf(1 - alpha)
    MoE = z_critical * SE  # Margin of Error
    CI = (mean - MoE, mean + MoE)  # Confidence Interval

    probability = None
    if range_start is not None and range_end is not None:
        ''' If you want to calculate z score from a normal distribution
        z_start = (range_start - proportion) / std_error
        z_end = (range_end - proportion) / std_error
        probability_within_range = stats.norm.cdf(z_end) - stats.norm.cdf(z_start)'''
        dist = stats.norm(mean, SE)
        probability = dist.cdf(range_end) - dist.cdf(range_start)

    return MoE, CI, probability


def t_distribution(mean, SE, cl, n, start, end):
    """
    Estimate using T Distribution
    :param mean: Sample mean (float)
    :param SE: Standard Error (float)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param n: Sample size (int)
    :param start: Start of the range for probability calculation (float or None)
    :param end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the value falling within the specified range (float or None)
    """
    df = n - 1  # Degree of Freedom
    dist = stats.t(df)
    alpha = (1 - cl) / 2  # Significance level
    t_critical = stats.t.ppf(1 - alpha, df)
    MoE = t_critical * SE  # Margin of Error
    CI = (mean - MoE, mean + MoE)  # Confidence Interval

    probability = None
    if start is not None and end is not None:
        t_start = (start - mean) / SE
        t_end = (end - mean) / SE
        probability = dist.cdf(t_end) - dist.cdf(t_start)

    return MoE, CI, probability


def calculate_mean_estimation(mean, n, std, is_sample_std=1, cl=0.95, start=None, end=None):
    """
    Estimate population parameters using the mean and standard deviation.
    :param mean: Sample mean (float)
    :param std: Standard deviation (float)
    :param is_sample_std: 1 for std and 0 for population std (bool)
    :param n: Sample size (int)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param start: Start of the range for probability calculation (float or None)
    :param end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the value falling within the specified range (float or None)
    """
    SE = std / math.sqrt(n)  # Standard error

    if n < 30 or is_sample_std:
        return t_distribution(mean, SE, cl, n, start, end)  # when sample size < 30 or σ is unknown.
    else:
        return z_distribution(mean, SE, cl, start, end)   # when sample size >= 30 or σ is known.


def calculate_proportion_estimation(p, n, cl=0.95, start=None, end=None):
    """
    Estimate population parameters using proportion.
    :param p: Sample proportion (float), where 0 <= p <= 1
    :param n: Sample size (int)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param start: Start of the range for probability calculation (float or None)
    :param end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the proportion falling within the specified range (float or None)
    """
    # Validate inputs
    is_valid, validation_message = validate_estimation_inputs(p, n)
    if not is_valid:
        raise ValueError(validation_message)

    SE = math.sqrt((p * (1 - p)) / n)

    return z_distribution(p, SE, cl, start, end)


# Example usage for mean estimation
sample_mean = 2.29
sample_std = .20
sample_size = 12
confidence_interval = 0.90
start = 2.1
end = 2.25

MoE, CI, probability = calculate_mean_estimation(sample_mean, sample_size, sample_std, 1, confidence_interval, start, end)
print(f"Mean Estimation - Margin of Error: {MoE:.2f}")
print(f"Mean Estimation - Confidence Interval: ({CI[0]:.2f}, {CI[1]:.2f})")
print(f"Mean Estimation - Probability within range [{start}, {end}]: {probability:.4f}")
print("\n")

sample_mean = 299720
population_std = 68650
sample_size = 1500
confidence_interval = 0.95
start = 290000
end = 300000

MoE, CI, probability = calculate_mean_estimation(sample_mean, sample_size, population_std, 0, confidence_interval, start, end)
print(f"Mean Estimation - Margin of Error: {MoE:.2f}")
print(f"Mean Estimation - Confidence Interval: ({CI[0]:.2f}, {CI[1]:.2f})")
print(f"Mean Estimation - Probability within range [{start}, {end}]: {probability:.4f}")


# Proportion estimation
sample_proportion = 0.44
sample_size = 1000
confidence_interval = 0.95
start = 0.45
end = 0.47

try:
    MoE, CI, probability = calculate_proportion_estimation(sample_proportion, sample_size, confidence_interval, start, end)
    print(f"Proportion Estimation - Margin of Error: {MoE:.4f}")
    print(f"Proportion Estimation - Confidence Interval: ({CI[0]:.4f}, {CI[1]:.4f})")
    print(f"Proportion Estimation - Probability within range [{start}, {end}]: {probability:.4f}")
except ValueError as e:
    print(f"Validation Error: {e}")
