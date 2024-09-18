import numpy as np
from scipy import stats

'''
P(Reject H₀ | H₀ is true) = α
P(Fail to reject H₀ | H₁ is true) = β
P(Reject H₀ | H₁ is true) = 1 - β

P(Reject the claim that the battery lasts 500 hours ∣ the battery does last 500 hours) = α
P(Fail to reject the claim that the battery lasts 500 hours ∣ the battery does not last 500 hours) = β
P(Reject the claim that the battery lasts 500 hours ∣ the battery does not last 500 hours) = 1 - β

In practice, you want the power to be at least 0.80, meaning there’s an 80% chance of correctly rejecting a false null hypothesis. If the power is too low, you may want to:
1. Increase the sample size.
2. Increase the significance level (though this increases the risk of a Type I error).
3. Reduce variability (if possible).
'''

def calculate_errors_power(mean_null, mean_alt, std, n, alpha=0.05, tails="two"):
    """
    Calculate Type I error, Type II error, and power of a hypothesis test for mean.
    :param mean_null: Mean under the null hypothesis
    :param mean_alt: Mean under the alternative hypothesis
    :param std: Standard deviation (assumed to be known)
    :param n: Sample size
    :param alpha: Significance level (Type I error probability)
    :param tails: 'two' for two-tailed test, 'left' for left-tailed, 'right' for right-tailed
    :return: Type I error (alpha), Type II error (beta), and power of the test
    """
    # Calculate the standard error
    SE = std / np.sqrt(n)

    # Z critical for alpha (type I error)
    if tails == "two":
        z_critical = stats.norm.ppf(1 - alpha / 2)
        critical_value_low = mean_null - z_critical * SE
        critical_value_high = mean_null + z_critical * SE
    elif tails == "right":
        z_critical = stats.norm.ppf(1 - alpha)
        critical_value_low = -np.inf  # No lower critical value for right-tailed test
        critical_value_high = mean_null + z_critical * SE
    elif tails == "left":
        z_critical = stats.norm.ppf(alpha)
        critical_value_low = mean_null + z_critical * SE
        critical_value_high = np.inf  # No upper critical value for left-tailed test

    # Type I Error is just alpha (pre-determined)
    type1_error = alpha

    # Type II Error (probability of failing to reject H0 when H0 is false)
    if tails == "two":
        beta = stats.norm.cdf(critical_value_high, loc=mean_alt, scale=SE) - stats.norm.cdf(critical_value_low, loc=mean_alt, scale=SE)
    elif tails == "right":
        beta = stats.norm.cdf(critical_value_high, loc=mean_alt, scale=SE)
    elif tails == "left":
        beta = 1 - stats.norm.cdf(critical_value_low, loc=mean_alt, scale=SE)

    # Power of the test (1 - β)
    power = 1 - beta

    return type1_error, beta, power


def calculate_proportion_errors_power(p_null, p_alt, n, alpha=0.05, tails="two"):
    """
    Calculate Type I error, Type II error, and power of a hypothesis test for proportion.
    :param p_null: Proportion under the null hypothesis
    :param p_alt: Proportion under the alternative hypothesis
    :param n: Sample size
    :param alpha: Significance level (Type I error probability)
    :param tails: 'two' for two-tailed test, 'left' for left-tailed, 'right' for right-tailed
    :return: Type I error (alpha), Type II error (beta), and power of the test
    """
    # Calculate the standard error for proportion
    SE_null = np.sqrt((p_null * (1 - p_null)) / n)

    # Z critical for alpha (Type I error)
    if tails == "two":
        z_critical = stats.norm.ppf(1 - alpha / 2)
        critical_value_low = p_null - z_critical * SE_null
        critical_value_high = p_null + z_critical * SE_null
    elif tails == "right":
        z_critical = stats.norm.ppf(1 - alpha)
        critical_value_low = -np.inf
        critical_value_high = p_null + z_critical * SE_null
    elif tails == "left":
        z_critical = stats.norm.ppf(alpha)
        critical_value_low = p_null + z_critical * SE_null
        critical_value_high = np.inf

    # Type I Error is just alpha (pre-determined)
    type1_error = alpha

    # Calculate SE for alternative proportion
    SE_alt = np.sqrt((p_alt * (1 - p_alt)) / n)

    # Type II Error (probability of failing to reject H0 when H0 is false)
    if tails == "two":
        beta = stats.norm.cdf(critical_value_high, loc=p_alt, scale=SE_alt) - stats.norm.cdf(critical_value_low, loc=p_alt, scale=SE_alt)
    elif tails == "right":
        beta = stats.norm.cdf(critical_value_high, loc=p_alt, scale=SE_alt)
    elif tails == "left":
        beta = 1 - stats.norm.cdf(critical_value_low, loc=p_alt, scale=SE_alt)

    # Power of the test (1 - β)
    power = 1 - beta

    return type1_error, beta, power


# Example inputs for mean hypothesis testing
mean_null = 15  # Null hypothesis mean
mean_alt = 15.5  # Alternative hypothesis mean (close to the null mean)
std = 0.5  # Standard deviation
n_mean = 10  # Sample size for mean
alpha = 0.05  # Significance level (Type I error)

# Example inputs for proportion hypothesis testing
p_null = 0.5  # Null hypothesis proportion
p_alt = 0.65  # Alternative hypothesis proportion (slightly higher)
n_proportion = 100  # Sample size for proportion

# Perform the calculations for mean (adjust 'tails' as 'left', 'right', or 'two')
type1_error_mean, beta_mean, power_mean = calculate_errors_power(mean_null, mean_alt, std, n_mean, alpha, tails="two")
print(f"Mean Test - Type I Error (Alpha): {type1_error_mean:.4f}")
print(f"Mean Test - Type II Error (Beta): {beta_mean:.4f}")
print(f"Mean Test - Power of the Test: {power_mean:.4f}")

# Perform the calculations for proportion (adjust 'tails' as 'left', 'right', or 'two')
type1_error_proportion, beta_proportion, power_proportion = calculate_proportion_errors_power(p_null, p_alt, n_proportion, alpha, tails="two")
print(f"Proportion Test - Type I Error (Alpha): {type1_error_proportion:.4f}")
print(f"Proportion Test - Type II Error (Beta): {beta_proportion:.4f}")
print(f"Proportion Test - Power of the Test: {power_proportion:.4f}")
