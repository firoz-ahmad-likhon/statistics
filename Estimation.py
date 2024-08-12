import scipy.stats as stats
import math


def validate_estimation_inputs(proportion):
    success = sample_size * proportion
    fail = sample_size * (1 - proportion)
    if success < 10 or fail < 10:
        return False, "For proportion estimation, success and fail must both be at least 10."

    return True, "Inputs are valid."


def calculate_mean_estimation(mean, std_dev, sample_size, confidence_level=0.95, range_start=None, range_end=None):

    std_error = std_dev / math.sqrt(sample_size)
    alpha = (1 - confidence_level) / 2

    if sample_size < 30:
        # Use t-distribution for small sample sizes
        degrees_of_freedom = sample_size - 1
        critical_value = stats.t.ppf(1 - alpha, df=degrees_of_freedom)
        distribution = stats.t(df=degrees_of_freedom)
    else:
        # Use Z-distribution for larger sample sizes
        critical_value = stats.norm.ppf(1 - alpha)
        distribution = stats.norm()

    margin_of_error = critical_value * std_error

    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    # Probability calculation within the specified range
    probability_within_range = None
    if range_start is not None and range_end is not None:
        z_start = (range_start - mean) / std_error
        z_end = (range_end - mean) / std_error
        probability_within_range = distribution.cdf(z_end) - distribution.cdf(z_start)

    return margin_of_error, confidence_interval, probability_within_range


def calculate_proportion_estimation(proportion, sample_size, confidence_level=0.95,
                                    range_start=None, range_end=None):
    # Validate inputs
    is_valid, validation_message = validate_estimation_inputs(proportion)
    if not is_valid:
        raise ValueError(validation_message)

    std_error = math.sqrt((proportion * (1 - proportion)) / sample_size)

    alpha = (1 - confidence_level) / 2
    critical_value = stats.norm.ppf(1 - alpha)
    distribution = stats.norm()

    margin_of_error = critical_value * std_error

    confidence_interval = (proportion - margin_of_error, proportion + margin_of_error)

    # Probability calculation within the specified range
    probability_within_range = None
    if range_start is not None and range_end is not None:
        z_start = (range_start - proportion) / std_error
        z_end = (range_end - proportion) / std_error

        probability_within_range = distribution.cdf(z_end) - distribution.cdf(z_start)

    return margin_of_error, confidence_interval, probability_within_range


# Example usage for mean estimation
mean = 100
std_dev = 15
sample_size = 25
confidence_level = 0.95
range_start = 95
range_end = 100

try:
    margin_of_error, confidence_interval, probability_within_range = calculate_mean_estimation(
        mean, std_dev, sample_size, confidence_level, range_start, range_end
    )
    print(f"Mean Estimation - Margin of Error: {margin_of_error:.2f}")
    print(f"Mean Estimation - Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")
    print(f"Mean Estimation - Probability within range [{range_start}, {range_end}]: {probability_within_range:.4f}")
except ValueError as e:
    print(f"Validation Error: {e}")

# Example usage for proportion estimation
proportion = 0.44
sample_size = 1000
confidence_level = 0.95
range_start = 0.40
range_end = 0.48

try:
    margin_of_error, confidence_interval, probability_within_range = calculate_proportion_estimation(
        proportion, sample_size=sample_size, confidence_level=confidence_level,
        range_start=range_start, range_end=range_end
    )
    print(f"Proportion Estimation - Margin of Error: {margin_of_error:.4f}")
    print(f"Proportion Estimation - Confidence Interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
    print(
        f"Proportion Estimation - Probability within range [{range_start}, {range_end}]: {probability_within_range:.4f}")
except ValueError as e:
    print(f"Validation Error: {e}")
