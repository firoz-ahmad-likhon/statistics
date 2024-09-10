import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Calculate Coefficient of determination(R^2)
r_squared = r_value ** 2

# Predict y values
y_pred = slope * x + intercept

# Calculate residuals (squared errors)
residuals = y - y_pred

mean_squared_error = np.mean(residuals ** 2)
root_mean_squared_error = np.sqrt(mean_squared_error)

# Plot using seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y, color='blue', label='Data points')
sns.lineplot(x=x, y=y_pred, color='black', label='Regression line')

# Plot residuals as vertical and horizontal lines
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]], color='red', linestyle='--', linewidth=1)
    plt.plot([x[i], x[i]], [y_pred[i], y[i]], color='white', linestyle='--', linewidth=1)

# Annotate with regression equation, R-value, R-squared, and mean squared error
plt.text(2, 10, f'y = {slope:.2f}x {"+" if intercept >= 0 else "-"} {abs(intercept):.2f}', fontsize=12, color='black', ha='left')
plt.text(2, 9.5, f'R (Correlation coefficient): {r_value:.2f}', fontsize=12, color='black', ha='left')
plt.text(2, 9, f'R^2 (Coefficient of determination): {r_squared:.2f}', fontsize=12, color='black', ha='left')
plt.text(2, 8.5, f'Mean Squared Error: {mean_squared_error:.2f}', fontsize=12, color='black', ha='left')
plt.text(2, 8, f'Root Mean Squared Error: {root_mean_squared_error:.2f}', fontsize=12, color='black', ha='left')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.grid(False)  # Remove gridlines
plt.tight_layout()  # Ensures tight layout to prevent overlap
plt.show()
