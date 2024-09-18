import numpy as np
from scipy import stats

'''
In practice, you want the power to be at least 0.80, meaning there’s an 80% chance of correctly rejecting a false null hypothesis. If the power is too low, you may want to:
1. Increase the sample size.
2. Increase the significance level (though this increases the risk of a Type I error).
3. Reduce variability (if possible).

Here’s a detailed example illustrating Type I error, Type II error, and the power of the test in the context of hypothesis testing:
Scenario: Testing a New Medication

A pharmaceutical company is developing a new medication that they claim lowers blood pressure by at least 10 mmHg on average. A clinical trial is conducted, and the company wants to test this claim.
Hypotheses:

    Null Hypothesis (H₀): The new medication lowers blood pressure by less than or equal to 10 mmHg on average (i.e., no significant improvement).
    Alternative Hypothesis (H₁): The new medication lowers blood pressure by more than 10 mmHg on average (i.e., significant improvement).

This is a right-tailed test, since we're testing whether the new medication leads to a greater effect than the baseline (10 mmHg).
Errors and Power in Hypothesis Testing:
1. Type I Error (False Positive):

    Definition: Rejecting the null hypothesis (H₀) when it is actually true.

    In this context: The trial results suggest that the new medication lowers blood pressure by more than 10 mmHg (i.e., reject H₀), but in reality, it does not. The company concludes the medication is effective when it is not.

    Significance Level (α): The probability of making a Type I error. If the significance level is set to 5% (α = 0.05), there is a 5% chance of incorrectly rejecting the null hypothesis and falsely concluding that the medication works.

    Example: Suppose the medication actually only lowers blood pressure by 9 mmHg on average, but due to random variation in the trial, the sample shows a mean decrease of 11 mmHg. This leads to incorrectly rejecting H₀ and concluding that the medication works, even though it does not.

2. Type II Error (False Negative):

    Definition: Failing to reject the null hypothesis (H₀) when it is false.

    In this context: The trial results suggest that the new medication does not lower blood pressure by more than 10 mmHg (i.e., fail to reject H₀), but in reality, it does. The company concludes that the medication is ineffective when it actually works.

    Probability (β): The probability of making a Type II error depends on the sample size, effect size, and standard deviation.

    Example: Suppose the medication actually lowers blood pressure by 12 mmHg, but the trial shows only a 9.8 mmHg decrease due to random variability. As a result, the company fails to reject H₀, missing out on a beneficial treatment.

3. Power of the Test (1 - β):

    Definition: The probability of correctly rejecting the null hypothesis (H₀) when it is false.
    In this context: The test correctly identifies that the new medication lowers blood pressure by more than 10 mmHg on average.
    Power depends on several factors:
        Sample size: Larger sample sizes increase the power of the test.
        Effect size: The larger the true effect (difference from 10 mmHg), the more likely the test will detect it.
        Significance level (α): A lower significance level reduces the Type I error but may also reduce power.
    Example: The actual effect of the medication is 12 mmHg, and the trial shows a decrease of 11.5 mmHg. The test correctly rejects H₀, concluding that the medication is effective.

Visualization of Errors and Power:

    Type I Error (α): The area in the right tail of the distribution under H₀ beyond the critical value, where we would reject H₀. This is the false positive region.
    Type II Error (β): The area under the distribution of H₁ that overlaps with the distribution of H₀, representing cases where we fail to reject H₀ even though H₁ is true.
    Power (1 - β): The area under the H₁ distribution beyond the critical value, where we correctly reject H₀.

Example Using Numbers:

Let’s say:

    The null hypothesis mean (H₀) is 10 mmHg.
    The alternative hypothesis mean (H₁) is 12 mmHg.
    The standard deviation is 1.5 mmHg.
    The sample size is 25.
    The significance level (α) is 0.05.

We calculate the Z-scores, critical values, and use these to determine the Type I error (α), Type II error (β), and power of the test.

In this case:

    Type I Error: The probability of rejecting H₀ when the medication doesn’t work (false positive).
    Type II Error: The probability of not rejecting H₀ when the medication actually works (false negative).
    Power: The probability of correctly detecting that the medication works.

Conclusion:

    Type I error: Rejects a true null hypothesis, leading to a false claim that the medication is effective.
    Type II error: Fails to reject a false null hypothesis, missing out on a treatment that actually works.
    Power: The ability of the test to detect the true effect when the medication is effective.
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
