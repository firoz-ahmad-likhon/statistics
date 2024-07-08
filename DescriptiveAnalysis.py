import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns

speed = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

print("-----Start Central Tendency-----")
mean = np.mean(speed)
median = np.median(speed)
mode = st.mode(speed)[0]
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("-----End Central Tendency-----\n\n")

print("-----Start Spread Analysis-----")
min = np.min(speed)
max = np.max(speed)
range = np.ptp(speed)
mid_range = range / 2
q25, q50, q75 = np.percentile(speed, [25, 50, 75])
iqr = q75 - q25
variance = np.var(speed)
std = np.std(speed)
mad_mean = np.mean(np.abs(speed - np.mean(speed)))
mad_median = np.median(np.abs(speed - np.median(speed)))
print('min', min)
print('max', max)
print('range', range)
print('mid range', mid_range)
print('25%', np.percentile(speed, 25))
print('50%', q50)
print('75%', q75)
print('iqr', iqr)
print('iqr', st.iqr(speed))
print('variance', variance)
print('standard deviation', std)
print('mad(mean)', mad_mean)
print('mad(median)', mad_median)
print('mad(median) using function', st.median_abs_deviation(speed))
print("-----End Spread Analysis-----\n\n")

print("-----Start Data Modeling-----")
zscore = st.zscore(speed)
loc = np.where(speed == 86)[0][0]
percentile_86 = st.percentileofscore(speed, 86)
ordinal_rank_86 = st.rankdata(speed, method='ordinal')[loc]
dense_rank_86 = st.rankdata(speed, method='dense')[loc]
zscore_86 = zscore[loc]
print('percentile of 86', percentile_86)
print('ordinal ranking of 86', ordinal_rank_86)
print('dense ranking of 86', dense_rank_86)

print('z-score of 86', zscore_86)
print("-----End Data Modeling-----\n\n")

print("-----Start Shape-----")
def shape_interpretation(skewness, kurtosis):
    # Interpret skewness
    if skewness == 0:
        skew_desc = "Symmetric distribution"
    elif 0 < skewness < 1:
        skew_desc = "Moderately right-skewed"
    elif skewness >= 1:
        skew_desc = "Highly right-skewed"
    elif -1 < skewness < 0:
        skew_desc = "Moderately left-skewed"
    else:
        skew_desc = "Highly left-skewed"

    # Interpret kurtosis
    if kurtosis == 3:
        kurt_desc = "Mesokurtic (normal distribution)"
    elif kurtosis > 3:
        kurt_desc = "Leptokurtic (heavy tails)"
    else:
        kurt_desc = "Platykurtic (light tails)"

    return skew_desc, kurt_desc

skewness = st.skew(speed)
kurt = st.kurtosis(speed)

skew_desc, kurt_desc = shape_interpretation(skewness, kurt)
print(f"Skewness: {skewness:.2f}, {skew_desc}")
print(f"Kurtosis: {kurt:.2f}", kurt_desc)
print("-----End Shape-----\n\n")

print("-----Start Outlier Detection-----")
# Function to remove outliers using IQR
def remove_outliers_iqr(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    return [val for val in data if lower_bound <= val <= upper_bound]

# Function to remove outliers using Z-score
def remove_outliers_zscore(data, threshold=3):
    z_scores = np.abs(st.zscore(data))
    return [val for key, val in enumerate(data) if z_scores[key] <= threshold]

# Function to remove outliers using modified Z-score
def remove_outliers_modified_zscore(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    return [val for key, val in enumerate(data) if np.abs(modified_z_scores[key]) <= threshold]

speed_iqr = remove_outliers_iqr(speed)
speed_zscore = remove_outliers_zscore(speed)
speed_modified_zscore = remove_outliers_modified_zscore(speed)
print("-----End Outlier Detection-----\n\n")

print("-----Start Visualizing-----")
# Set up Seaborn style
sns.set(style="whitegrid")

# Create a figure with subplots
fig, axs = plt.subplots(3, 2, figsize=(16, 18))

# Plot 1: Annotations for Central Tendency
axs[0, 0].hist(speed, bins=5, edgecolor='black', alpha=0.7, color='skyblue')
axs[0, 0].axvline(x=mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
axs[0, 0].axvline(x=median, color='green', linestyle='--', linewidth=2, label=f'Median: {median}')
axs[0, 0].axvline(x=mode, color='purple', linestyle='--', linewidth=2, label=f'Mode: {mode}')
axs[0, 0].set_title('Central Tendency Measures')
axs[0, 0].set_xlabel('Speed')
axs[0, 0].set_ylabel('Frequency')
axs[0, 0].legend()

# Plot 2: Annotations for Spread Analysis
axs[0, 1].boxplot(speed, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
axs[0, 1].axvline(x=min, color='blue', linestyle='--', linewidth=2, label=f'Min: {min}')
axs[0, 1].axvline(x=max, color='orange', linestyle='--', linewidth=2, label=f'Max: {max}')
axs[0, 1].axvline(x=q25, color='magenta', linestyle='--', linewidth=2, label=f'Q1: {q25}')
axs[0, 1].axvline(x=q50, color='brown', linestyle='--', linewidth=2, label=f'Median: {q50}')
axs[0, 1].axvline(x=q75, color='cyan', linestyle='--', linewidth=2, label=f'Q3: {q75}')
axs[0, 1].set_title('Spread Analysis Measures')
axs[0, 1].set_xlabel('Speed')
axs[0, 1].legend()

# Plot 3: Annotations for MAD and Variance
axs[1, 0].hist(speed, bins=5, edgecolor='black', alpha=0.7, color='skyblue')
axs[1, 0].axvline(x=mean + mad_mean, color='red', linestyle='--', linewidth=2,
                  label=f'Mean + MAD(mean): {mean + mad_mean:.2f}')
axs[1, 0].axvline(x=mean - mad_mean, color='red', linestyle='--', linewidth=2,
                  label=f'Mean - MAD(mean): {mean - mad_mean:.2f}')
axs[1, 0].axvline(x=median + mad_median, color='green', linestyle='--', linewidth=2,
                  label=f'Median + MAD(median): {median + mad_median:.2f}')
axs[1, 0].axvline(x=median - mad_median, color='green', linestyle='--', linewidth=2,
                  label=f'Median - MAD(median): {median - mad_median:.2f}')
axs[1, 0].set_title('MAD Measures')
axs[1, 0].set_xlabel('Speed')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].legend()

# Plot 4: Annotations for Data Modeling
axs[1, 1].hist(speed, bins=5, edgecolor='black', alpha=0.7, color='skyblue')
axs[1, 1].annotate(f'Percentile of 86: {percentile_86:.2f}%', xy=(86, 2), xytext=(100, 5),
                   arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
axs[1, 1].annotate(f'Ordinal Rank of 86: {ordinal_rank_86}', xy=(86, 3), xytext=(100, 6),
                   arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
axs[1, 1].annotate(f'Dense Rank of 86: {dense_rank_86}', xy=(86, 4), xytext=(100, 7),
                   arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
axs[1, 1].annotate(f'Z-score of 86: {zscore_86:.2f}', xy=(86, 5), xytext=(100, 8),
                   arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
axs[1, 1].set_title('Data Modeling Measures')
axs[1, 1].set_xlabel('Speed')
axs[1, 1].set_ylabel('Frequency')

# Plot 5: Bell Curve with Standard Deviation and Empirical Rules
sns.histplot(speed, kde=True, stat='density', linewidth=0, color='skyblue', ax=axs[2, 0])
xmin, xmax = axs[2, 0].get_xlim()
x = np.linspace(xmin, xmax, 100)
pdf = st.norm.pdf(x, mean, std)
axs[2, 0].plot(x, pdf, linewidth=2, color='red', label='Normal Distribution')
axs[2, 0].axvline(mean, color='k', linestyle='--', linewidth=1, label='Mean')
axs[2, 0].axvline(mean + std, color='blue', linestyle='--', linewidth=1, label='Mean + Std')
axs[2, 0].axvline(mean - std, color='blue', linestyle='--', linewidth=1, label='Mean - Std')
axs[2, 0].axvline(mean + 2*std, color='green', linestyle='--', linewidth=1, label='Mean + 2*Std')
axs[2, 0].axvline(mean - 2*std, color='green', linestyle='--', linewidth=1, label='Mean - 2*Std')
axs[2, 0].axvline(mean + 3*std, color='purple', linestyle='--', linewidth=1, label='Mean + 3*Std')
axs[2, 0].axvline(mean - 3*std, color='purple', linestyle='--', linewidth=1, label='Mean - 3*Std')
axs[2, 0].set_title('Bell Curve with Standard Deviation and Empirical Rules')
axs[2, 0].set_xlabel('Speed')
axs[2, 0].set_ylabel('Density')
axs[2, 0].legend()

# Plot 6: Bell Curve After Outlier Removal using Modified Z-score
sns.histplot(speed_modified_zscore, kde=True, stat='density', linewidth=0, color='skyblue', ax=axs[2, 1])
xmin, xmax = axs[2, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
pdf = st.norm.pdf(x, np.mean(speed_modified_zscore),np.std(speed_modified_zscore))
axs[2, 1].plot(x, pdf, linewidth=2, color='red', label='Normal Distribution')
axs[2, 1].set_title('Bell Curve After Outlier Removal')
axs[2, 1].set_xlabel('Speed')
axs[2, 1].set_ylabel('Density')
axs[2, 1].legend()

# Adjust layout and display plot
plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing between subplots
plt.show()
print("-----End Visualizing-----")
