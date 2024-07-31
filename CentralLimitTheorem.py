''' When data is continuous, sample_size >= 30 and population_std is known then
    Mean of all sample_mean = population_mean,
    SE = population_std / sqrt(sample_size),
    Sample_mean follows normal distribution

    When data is categorical, n * p >= 5 and n * (1 - p) >= 5 then
    SE = sqrt(p * (1 - p) / n)
    ME = z * SE
    Sample proportion follows normal distribution
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
population_mean = 50
sample_size = 40
num_samples = 1000

# Generate population data (exponential distribution for skewed data)
population_data = np.random.exponential(scale=population_mean, size=10000)

print("-----Start statistics for population from data-----")
population_mean_actual = np.mean(population_data)
population_std_actual = np.std(population_data)

print(f'Population Mean: {population_mean_actual:.2f}')
print(f'Population Std Dev: {population_std_actual:.2f}')
print("-----End statistics for population from data-----\n\n")

print("-----Start statistics for sample mean-----")
# Collect sample means
sample_means = []
for _ in range(num_samples):
    sample = np.random.choice(population_data, sample_size)
    sample_means.append(np.mean(sample))

# Calculate statistics for sample means
mean_of_sample_means = np.mean(sample_means)
std_of_sample_means = np.std(sample_means)

print(f'Mean of Sample Means: {mean_of_sample_means:.2f}')
print(f'Std Dev of Sample Means: {std_of_sample_means:.2f}')
print("-----End statistics for sample mean-----\n\n")

print("-----Start statistics by CLT-----")
# Calculate statistics
clt_mean = population_mean_actual
clt_std = population_std_actual / np.sqrt(sample_size)

print(f'Mean by CLT: {clt_mean:.2f}')
print(f'Std Dev by CLT: {clt_std:.2f}')
print("-----End statistics by CLT-----\n\n")

print("-----Start Visualizing-----")
# Plot the population distribution
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.hist(population_data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
x_pop = np.linspace(min(population_data), max(population_data), 100)
pdf_pop = norm.pdf(x_pop, population_mean_actual, population_std_actual)
plt.plot(x_pop, pdf_pop, 'k', linewidth=2, label='PDF')
plt.title('Skewed Population Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend([f'Mean: {population_mean_actual:.2f}\nStd Dev: {population_std_actual:.2f}'])

# Plot the sampling distribution of the sample mean
plt.subplot(1, 2, 2)
plt.hist(sample_means, bins=30, density=True, alpha=0.6, color='lightgreen', edgecolor='black')
plt.title('Sampling Distribution of the Sample Mean')
plt.xlabel('Sample Mean')
plt.ylabel('Density')

# Overlay the normal distribution
x_sample = np.linspace(min(sample_means), max(sample_means), 100)
pdf_sample = norm.pdf(x_sample, mean_of_sample_means, std_of_sample_means)
plt.plot(x_sample, pdf_sample, 'k', linewidth=2, label='Normal PDF')
plt.legend([f'Mean: {mean_of_sample_means:.2f}\nStd Dev: {std_of_sample_means:.2f}'])

plt.tight_layout()
plt.show()
print("-----End Visualizing-----")
