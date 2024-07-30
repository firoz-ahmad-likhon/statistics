import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

speed = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

# Create a normal distribution
dist = norm(np.mean(speed), np.std(speed))

print("-----Start calculating probability under points-----")
# Calculate the CDF values at 86 and 99
cdf_86 = dist.cdf(86)
cdf_99 = dist.cdf(99)

# Calculate the probabilities
prob_between_86_and_99 = cdf_99 - cdf_86
prob_greater_than_99 = 1 - cdf_99
prob_less_than_86 = cdf_86

print(f"Probability of being between 86 and 99: {prob_between_86_and_99:.2f}")
print(f"Probability of being greater than 99: {prob_greater_than_99:.2f}")
print(f"Probability of being less than 86: {prob_less_than_86:.2f}")
print("-----End calculating probability under points-----\n\n")

print("-----Start calculating probability under percentiles-----")
# ppf is the inverse of cdf: it takes a probability and returns the corresponding value
# Find the values at the 34th and 84th percentiles
point_of_34 = dist.ppf(0.34)
point_of_84 = dist.ppf(0.84)

# Calculate the CDF values at these points
cdf_point_of_34 = dist.cdf(point_of_34)
cdf_point_of_84 = dist.cdf(point_of_84)

# Calculate the probabilities
prob_between_point_of_34_and_point_of_84 = cdf_point_of_84 - cdf_point_of_34
prob_greater_than_point_of_84 = 1 - cdf_point_of_84
prob_less_than_point_of_34 = cdf_point_of_34

print(f"Probability of being between 34th and 84th percentile: {prob_between_point_of_34_and_point_of_84:.2f}")
print(f"Probability of being greater than 84th percentile: {prob_greater_than_point_of_84:.2f}")
print(f"Probability of being less than 34th percentile: {prob_less_than_point_of_34:.2f}")
print("-----End calculating probability under percentiles-----\n\n")

print("-----Start Visualizing-----")
# Define the range of values
x_values = np.linspace(min(speed) - 10, max(speed) + 20, 1000)

# Calculate the PDF values
pdf_values = dist.pdf(x_values)

# Plot the PDF and highlight areas
plt.figure(figsize=(12, 12))

# Plot 1: PDF, CDF with highlighted areas for 86 and 99
plt.subplot(1, 2, 1)
sns.lineplot(x=x_values, y=pdf_values, color='blue', label='PDF')
plt.fill_between(np.linspace(86, 99, 1000), dist.pdf(np.linspace(86, 99, 1000)), color='orange', alpha=0.3, label=f'P(86 ≤ X ≤ 99) = {prob_between_86_and_99:.2f}')
plt.fill_between(np.linspace(99, max(speed)+10, 1000), dist.pdf(np.linspace(99, max(speed)+10, 1000)), color='green', alpha=0.3, label=f'P(X > 99) = {prob_greater_than_99:.2f}')
plt.fill_between(np.linspace(min(speed)-10, 86, 1000), dist.pdf(np.linspace(min(speed)-10, 86, 1000)), color='red', alpha=0.3, label=f'P(X < 86) = {prob_less_than_86:.2f}')
plt.scatter([86, 99], [dist.pdf(86), dist.pdf(99)], color='black')
plt.text(86, dist.pdf(86), f'86\nPDF: {dist.pdf(86):.2f}\n CDF: {cdf_86: .2f}', color='black', ha='right', va='bottom')
plt.text(99, dist.pdf(99), f'99\nPDF: {dist.pdf(99):.2f}\n CDF: {cdf_99: .2f}', color='black', ha='left', va='bottom')
plt.text(92.5, 0.02, f'{prob_between_86_and_99:.2f}', color='orange', ha='center')
plt.text(105, 0.005, f'{prob_greater_than_99:.2f}', color='green', ha='center')
plt.text(75, 0.005, f'{prob_less_than_86:.2f}', color='red', ha='center')
plt.xlabel('Speed')
plt.ylabel('PDF')
plt.title('Probability Under Points')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()

# Plot 2: PDF, PPF with highlighted areas for 34% and 84%
plt.subplot(1, 2, 2)
sns.lineplot(x=x_values, y=pdf_values, color='blue', label='PDF')
plt.fill_between(np.linspace(point_of_34, point_of_84, 1000), dist.pdf(np.linspace(point_of_34, point_of_84, 1000)), color='orange', alpha=0.3, label=f'P(34th ≤ X ≤ 84th) = {prob_between_point_of_34_and_point_of_84:.2f}')
plt.fill_between(np.linspace(point_of_84, max(x_values), 1000), dist.pdf(np.linspace(point_of_84, max(x_values), 1000)), color='green', alpha=0.3, label=f'P(X > 84th) = {prob_greater_than_point_of_84:.2f}')
plt.fill_between(np.linspace(min(x_values), point_of_34, 1000), dist.pdf(np.linspace(min(x_values), point_of_34, 1000)), color='red', alpha=0.3, label=f'P(X < 34th) = {prob_less_than_point_of_34:.2f}')
plt.scatter([point_of_34, point_of_84], [dist.pdf(point_of_34), dist.pdf(point_of_84)], color='black')
plt.text(point_of_34, dist.pdf(point_of_34), f'34th\nPDF: {dist.pdf(point_of_34):.2f}\n CDF: {cdf_point_of_34: .2f}', color='black', ha='right', va='bottom')
plt.text(point_of_84, dist.pdf(point_of_84), f'84th\nPDF: {dist.pdf(point_of_84):.2f}\n CDF: {cdf_point_of_84: .2f}', color='black', ha='left', va='bottom')
plt.text(np.mean(x_values) - 7, 0.02, f'{prob_between_point_of_34_and_point_of_84:.2f}', color='orange', ha='center')
plt.text(np.max(x_values) - 26, 0.005, f'{prob_greater_than_point_of_84:.2f}', color='green', ha='center')
plt.text(np.min(x_values) + 10, 0.005, f'{prob_less_than_point_of_34:.2f}', color='red', ha='center')
plt.xlabel('Speed')
plt.ylabel('PDF')
plt.title('Probability Under Percentiles')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()

plt.show()
print("-----End Visualizing-----")