import scipy.stats as stats
import math


def validate_proportion_inputs(p, n):
    """
    Validate inputs for proportion hypothesis testing.
    :param p: Sample proportion (float)
    :param n: Sample size (int)
    :return: Tuple of boolean and validation message
    """
    if n * p < 10 or n * (1 - p) < 10:
        return False, "For proportion estimation, succeeded = np and failed = n(1-p) must both be at least 10."
    return True, "Inputs are valid."


def calculate_standard_error_mean(std_dev, n):
    """
    Calculate the standard error for the mean.
    :param std_dev: Standard deviation (float)
    :param n: Sample size (int)
    :return: Standard error (float)
    """
    return std_dev / math.sqrt(n)


def calculate_standard_error_proportion(p, n):
    """
    Calculate the standard error for a proportion.
    :param p: Sample proportion (float)
    :param n: Sample size (int)
    :return: Standard error (float)
    """
    return math.sqrt((p * (1 - p)) / n)


def calculate_z_score(sample_stat, population_mean, se):
    """
    Calculate the z-score for a hypothesis test.
    :param sample_stat: Sample statistic (mean or proportion) (float)
    :param population_mean: Hypothesized population mean or proportion (float)
    :param se: Standard error (float)
    :return: Z-score (float)
    """
    return (sample_stat - population_mean) / se


def calculate_t_score(sample_mean, population_mean, se):
    """
    Calculate the t-score for a hypothesis test.
    :param sample_mean: Sample mean (float)
    :param population_mean: Hypothesized population mean (float)
    :param se: Standard error (float)
    :return: T-score (float)
    """
    return (sample_mean - population_mean) / se


def calculate_p_value(z_score, tail_type='two', df=None):
    """
    Calculate the p-value for a hypothesis test.
    :param z_score: Z-score or T-score (float)
    :param tail_type: Type of test ('two', 'left', 'right')
    :param df: Degrees of freedom (int), used for T-distribution
    :return: p-value (float)
    """
    if tail_type == 'two':
        if df is None:
            return 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            return 2 * (1 - stats.t.cdf(abs(z_score), df))
    elif tail_type == 'left':
        if df is None:
            return stats.norm.cdf(z_score)
        else:
            return stats.t.cdf(z_score, df)
    elif tail_type == 'right':
        if df is None:
            return 1 - stats.norm.cdf(z_score)
        else:
            return 1 - stats.t.cdf(z_score, df)
    else:
        raise ValueError("Invalid tail type. Choose 'two', 'left', or 'right'.")


def hypothesis_test_mean(sample_mean, population_mean, std_dev, n, alpha, tail_type='two', is_population_std=True):
    """
    Perform a hypothesis test for the mean.
    :param sample_mean: Sample mean (float)
    :param population_mean: Hypothesized population mean (float)
    :param std_dev: Population or Sample standard deviation (float)
    :param n: Sample size (int)
    :param alpha: Significance level (float)
    :param tail_type: Type of test ('two', 'left', 'right')
    :param is_population_std: Whether the standard deviation is of the population (True) or sample (False)
    :return: Tuple containing the test statistic (Z or T), p-value, and decision
    """
    se = calculate_standard_error_mean(std_dev, n)

    if n >= 30 or is_population_std:
        # Z-Test
        test_stat = calculate_z_score(sample_mean, population_mean, se)
        p_value = calculate_p_value(test_stat, tail_type)
    else:
        # T-Test
        test_stat = calculate_t_score(sample_mean, population_mean, se)
        df = n - 1  # Degrees of freedom for T-test
        p_value = calculate_p_value(test_stat, tail_type, df)

    decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"
    return test_stat, p_value, decision


def hypothesis_test_proportion(sample_proportion, population_proportion, n, alpha, tail_type='two'):
    """
    Perform a hypothesis test for the proportion.
    :param sample_proportion: Sample proportion (float)
    :param population_proportion: Hypothesized population proportion (float)
    :param n: Sample size (int)
    :param alpha: Significance level (float)
    :param tail_type: Type of test ('two', 'left', 'right')
    :return: Tuple containing the z-score, p-value, and decision
    """
    is_valid, validation_message = validate_proportion_inputs(population_proportion, n)
    if not is_valid:
        raise ValueError(validation_message)

    se = calculate_standard_error_proportion(population_proportion, n)
    z_score = calculate_z_score(sample_proportion, population_proportion, se)
    p_value = calculate_p_value(z_score, tail_type)
    decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"

    return z_score, p_value, decision


# Example Usage
# Z-Test or T-Test for Mean
sample_mean = 17  # Sample mean height
population_mean = 15  # Hypothesized population mean
std_dev = 0.5  # Standard deviation (known population std)
n = 10  # Sample size
alpha = 0.05  # Significance level

# Z-Test for Mean
test_stat, p_value, decision = hypothesis_test_mean(sample_mean, population_mean, std_dev, n, alpha, 'right',
                                                    is_population_std=True)
print(f"Mean Test (Z-Test) - Test Statistic: {test_stat:.2f}, p-value: {p_value:.4f}, Decision: {decision}")

# T-Test for Mean
test_stat, p_value, decision = hypothesis_test_mean(sample_mean, population_mean, std_dev, n, alpha, 'right',
                                                    is_population_std=False)
print(f"Mean Test (T-Test) - Test Statistic: {test_stat:.2f}, p-value: {p_value:.4f}, Decision: {decision}")

# Two-tailed Z-Test for Mean
sample_mean = 350  # Sample mean height
population_mean = 355  # Hypothesized population mean
std_dev = 8  # Standard deviation (known population std)
n = 30  # Sample size

# Z-Test for Mean
test_stat, p_value, decision = hypothesis_test_mean(sample_mean, population_mean, std_dev, n, alpha, 'two',
                                                    is_population_std=True)

print(f"Two tail Mean Test (Z-Test) - Test Statistic: {test_stat:.2f}, p-value: {p_value:.4f}, Decision: {decision}")

# Z-Test for Proportion
sample_proportion = 0.44  # Sample proportion
population_proportion = 0.5  # Hypothesized population proportion
n = 1000  # Sample size
alpha = 0.05  # Significance level

z_score, p_value, decision = hypothesis_test_proportion(sample_proportion, population_proportion, n, alpha, 'two')
print(f"Proportion Test - Z-score: {z_score:.2f}, p-value: {p_value:.4f}, Decision: {decision}")
