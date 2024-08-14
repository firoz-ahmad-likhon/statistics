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
    :param mean: Mean of the population or sample (float)
    :param SE: Standard Error (float)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param range_start: Start of the range for probability calculation (float or None)
    :param range_end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the value falling within the specified range (float or None)
    """
    alpha = (1 - cl) / 2
    z_critical = stats.norm.ppf(1 - alpha)

    MoE = z_critical * SE
    CI = (mean - MoE, mean + MoE)

    probability = None
    if range_start is not None and range_end is not None:
        ''' If you want to calculate z score from a normal distribution
        z_start = (range_start - proportion) / std_error
        z_end = (range_end - proportion) / std_error
        probability_within_range = stats.norm.cdf(z_end) - stats.norm.cdf(z_start)'''
        dist = stats.norm(mean, SE)
        probability = dist.cdf(range_end) - dist.cdf(range_start)

    return MoE, CI, probability


def t_distribution(mean, SE, cl, size, range_start, range_end):
    """
    Estimate using T Distribution.
    :param mean: Mean of the population or sample (float)
    :param SE: Standard Error (float)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param size: Sample size (int)
    :param range_start: Start of the range for probability calculation (float or None)
    :param range_end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the value falling within the specified range (float or None)
    """
    df = size - 1
    alpha = (1 - cl) / 2
    t_critical = stats.t.ppf(1 - alpha, df)
    dist = stats.t(df)

    MoE = t_critical * SE
    CI = (mean - MoE, mean + MoE)

    probability = None
    if range_start is not None and range_end is not None:
        z_start = (range_start - mean) / SE
        z_end = (range_end - mean) / SE
        probability = dist.cdf(z_end) - dist.cdf(z_start)

    # Calculate probability directly using the T distribution
    probability_1 = dist.cdf(range_end) - dist.cdf(range_start)

    print(probability, probability_1)

    return MoE, CI, probability


def calculate_mean_estimation(mean, std, n, cl=0.95, start=None, end=None):
    """
    Estimate population parameters using the mean and standard deviation.
    :param mean: Population mean (float)
    :param std: Population standard deviation (float)
    :param n: Sample size (int)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param start: Start of the range for probability calculation (float or None)
    :param end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the value falling within the specified range (float or None)
    """
    SE = std / math.sqrt(n)

    if n < 30:
        return t_distribution(mean, SE, cl, n, start, end)
    else:
        return z_distribution(mean, SE, cl, start, end)


def calculate_proportion_estimation(p, n, cl=0.95, start=None, end=None):
    """
    Estimate population parameters using proportion.
    :param p: Population proportion (float), where 0 <= p <= 1
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
mean = 2.29
std = .20
n = 12
cl = 0.90
start = 2.1
end = 2.25

try:
    MoE, CI, probability = calculate_mean_estimation(mean, std, n, cl, start, end)
    print(f"Mean Estimation - Margin of Error: {MoE:.2f}")
    print(f"Mean Estimation - Confidence Interval: ({CI[0]:.2f}, {CI[1]:.2f})")
    print(f"Mean Estimation - Probability within range [{start}, {end}]: {probability:.4f}")
except ValueError as e:
    print(f"Validation Error: {e}")

# Proportion estimation
p = 0.44
n = 1000
cl = 0.95
start = 0.45
end = 0.47

try:
    MoE, CI, probability = calculate_proportion_estimation(p, n, cl, start, end)
    print(f"Proportion Estimation - Margin of Error: {MoE:.4f}")
    print(f"Proportion Estimation - Confidence Interval: ({CI[0]:.4f}, {CI[1]:.4f})")
    print(f"Proportion Estimation - Probability within range [{start}, {end}]: {probability:.4f}")
except ValueError as e:
    print(f"Validation Error: {e}")
