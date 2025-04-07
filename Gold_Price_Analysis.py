# %% [markdown]
# # 📊 **Econophysics: Gold Price Analysis**
# This notebook presents an analysis of gold price data using Econophysics techniques. The workflow includes:
# - Calculating price differences and normalized differences
# - Plotting probability density functions (PDF) using Freedman-Diaconis bins and Gaussian KDE
# - Analyzing ordinal patterns and calculating permutation entropy
# - Visualizing conditional probabilities with a transition matrix heatmap
# 
# 🔥 **Key Steps**:
# - Reading the gold price data
# - Computing and plotting normalized differences
# - Ordinal pattern and entropy analysis
# - Heatmap visualization of transition matrices

# %%
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import itertools
from collections import Counter
import math
import pandas as pd
from collections import defaultdict
import seaborn as sns

# Step 1: Read the file
def read_file(file_path):
    """
    Load the gold price data from the file.

    Args:
        file_path (str): Path to the file containing gold price data.

    Returns:
        np.ndarray: An array of prices.
    """
    prices = np.loadtxt(file_path)
    return prices

# %%
def preprocess_gold_data(prices, window_size=10):
    """
    Preprocesses Gold price data to remove microstructure noise.

    Parameters:
    - file_path: str, path to the Gold price data file.
    - window_size: int, size of the moving average window.

    Returns:
    - x0: numpy.ndarray, standardized differences of the moving average.
    - tau: list, range of lag values for analysis.
    - M1_a: numpy.ndarray, measure 1 computed from kernel density estimation.
    - M2_a: numpy.ndarray, measure 2 computed from kernel density estimation.
    """
    # Step 2: Compute the moving average using convolution
    moving_avg = np.convolve(prices, np.ones(window_size)/window_size, mode='valid')

    # Replace NA values at the start with the first non-NA value (if any)
    moving_avg = np.concatenate(([prices[0]], moving_avg))

    # Calculate the difference of the moving average
    diff = np.diff(moving_avg)

    # Standardizing the data
    x0 = (diff - np.nanmean(diff)) / np.nanstd(diff)

    # Remove non-finite values
    x0 = x0[np.isfinite(x0)]

    # Parameters for kernel density estimation
    bw = 0.2
    na = 6
    tau = np.arange(1, 11)  # Range of tau from 1 to 10
    avec = np.linspace(np.min(x0), np.max(x0), na)

    # Scaling factor for kernel density estimation
    SF = 1 / (bw * np.sqrt(2 * np.pi))

    # Initialize result vectors for M1 and M2
    M1_a = np.zeros(len(tau))
    M2_a = np.zeros(len(tau))

    # Loop over tau to compute M1 and M2
    for i in range(len(tau)):
        # Calculate differences and their squares with lag tau[i]
        dx = np.diff(x0, n=tau[i])
        dx2 = dx ** 2
        nx = len(dx)

        # Initialize kernel matrix
        Kmat = np.zeros((na, nx))

        # Fill the kernel matrix for each j
        for j in range(nx):
            Kmat[:, j] = SF * np.exp(-0.5 * ((x0[j] - avec) ** 2) / (bw ** 2))

        # Sum of weights for the last row (kernel sum for na-th point)
        Ksum = np.sum(Kmat[na - 1, :])

        # Compute M1 and M2 using the kernel sum
        if Ksum != 0:
            M1_a[i] = np.sum(Kmat[na - 1, :] * dx) / Ksum
            M2_a[i] = np.sum(Kmat[na - 1, :] * dx2) / Ksum
        else:
            M1_a[i] = np.nan
            M2_a[i] = np.nan

    # Replace any remaining non-finite values in M1_a and M2_a
    M1_a[np.isnan(M1_a)] = 0
    M2_a[np.isnan(M2_a)] = 0
    # Plotting the results and save to a PDF file
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(tau, M1_a / tau, 'b-', label=r'M$^{(1)}(0, \tau)$ / $\tau$')
    plt.scatter(tau, M1_a / tau, color='red')
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'M$^{(1)}(0, \tau)$ / $\tau$')
    plt.ylim([np.min(M1_a / tau), np.max(np.abs(M1_a / tau))])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(tau, M2_a / tau, 'b-', label=r'M$^{(2)}(0, \tau)$ / $\tau$')
    plt.scatter(tau, M2_a / tau, color='red')
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'M$^{(2)}(0, \tau)$ / $\tau$')
    plt.ylim([0, np.max(M2_a / tau)])
    plt.legend()

    plt.tight_layout()
    plt.show()
    return np.array([np.mean(prices[i:i + window_size]) for i in range(len(prices) - window_size + 1)])

# %%
# Step 2: Calculate normalized price differences
def calculate_normalized_price_diff(prices):
    """
    Calculate the normalized price difference.

    Args:
        prices (np.ndarray): Array of price values.

    Returns:
        np.ndarray: Price differences.
        np.ndarray: Normalized price differences.
    """
    # Calculate price difference (x)
    price_diff = np.diff(prices)

    # Standard deviation of price differences
    sd_price_diff = np.std(price_diff)

    # Normalize the differences (y = x/sd(x))
    normalized_diff = price_diff / sd_price_diff

    return price_diff, normalized_diff
# Plot normalized difference y(N)
def plot_y_N(normalized_diff):
    """
    Plot the normalized differences y(N).

    Args:
        normalized_diff (np.ndarray): Normalized price differences.
    """
    plt.plot(np.arange(len(normalized_diff)), normalized_diff, label='Normalized y(N)', color='blue')
    plt.title('Normalized Difference y(N)')
    plt.xlabel('N (Index)')
    plt.ylabel('y(N)')
    plt.legend()
    plt.show()

# %%
# Step 4: Calculate PDF with Freedman-Diaconis Rule
def calculate_freedman_diaconis_bins(normalized_diff):
    """
    Calculate the number of bins using Freedman-Diaconis Rule.

    Args:
        normalized_diff (np.ndarray): Normalized price differences.

    Returns:
        int: Number of bins.
    """
    q1 = np.percentile(normalized_diff, 25)
    q3 = np.percentile(normalized_diff, 75)
    iqr = q3 - q1

    bin_width = 2 * iqr * len(normalized_diff) ** (-1/3)
    num_bins = int(np.ceil((normalized_diff.max() - normalized_diff.min()) / bin_width))

    return num_bins

def calculate_pdf_with_freedman_diaconis(normalized_diff, a = 1, b = 3.9):
    """
    Calculate and plot the PDF using the Freedman-Diaconis rule.

    Args:
        normalized_diff (np.ndarray): Normalized price differences.
        a (float): Parameter a for the fit function.
        b (float): Parameter b for the fit function.
    """
    num_bins = calculate_freedman_diaconis_bins(normalized_diff)
    hist, bin_edges = np.histogram(normalized_diff, bins=num_bins, density=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filter out non-positive bin centers
    valid_indices = bin_centers > 0
    bin_centers_filtered = bin_centers[valid_indices]
    hist_filtered = hist[valid_indices]

    # Define the manual fit function
    def fit_function(y):
        return a * (1 / (y ** b))

    # Generate fitted data for the range from 1 to 4
    y_fit = np.linspace(1,20, 1000)
    fitted_values = fit_function(y_fit)

    # Plotting
    plt.plot(bin_centers, hist, label=f'PDF with {num_bins} bins (Freedman-Diaconis)', color='green')
    plt.plot(y_fit, fitted_values, label=f'Fitted function: (1/y^{b:.2f})', color='red')

    # Add text annotation for the fit function
    plt.text(1.5, max(fitted_values)*0.8, f'Fit: (1/y^{b:.2f})', fontsize=12, color='red')

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Probability Density Function (PDF) - Log-Log Scale')
    plt.xlabel('Data (log scale)')
    plt.ylabel('Probability Density (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# %%
# Step 5: Calculate Ordinal Patterns and Permutation Entropy
def ordinal_pattern(prices, D):
    """
    Calculate ordinal patterns for the time series.

    Args:
        prices (np.ndarray): Time series data.
        D (int): Embedding dimension.

    Returns:
        list: Valid ordinal patterns.
        dict: Frequency of each ordinal pattern.
    """
    n = len(prices)
    if D > n:
        raise ValueError("Embedding dimension D must be smaller than the length of the time series.")

    windows = np.lib.stride_tricks.sliding_window_view(prices, D)

    valid_patterns = []
    for window in windows:
        if len(set(window)) != D:
            continue
        sorted_indices = np.argsort(window)
        valid_patterns.append(tuple(sorted_indices))

    pattern_counts = Counter(valid_patterns)
    return valid_patterns, pattern_counts
# Permutation Entropy calculation
def calculate_permutation_entropy(pattern_counts, D):
    total_patterns = sum(pattern_counts.values())
    if total_patterns == 0:
        return 0

    probabilities = np.array([count / total_patterns for count in pattern_counts.values()])
    entropy = -np.sum(probabilities * np.log(probabilities))
    max_entropy = np.log(math.factorial(D))

    return entropy / max_entropy

# %%
# Step 6: Calculate conditional probabilities and transition matrix
def calculate_conditional_probabilities_and_matrix(valid_patterns, D):
    """
    Calculates the conditional probabilities and transition matrix.

    Args:
        valid_patterns (list): List of valid ordinal patterns.
        D (int): Embedding dimension.

    Returns:
        tuple: conditional_probabilities (dict), transition_matrix (ndarray), unique_patterns (list)
    """
    num_patterns = math.factorial(D)
    data = {'Current': valid_patterns[:-1], 'Next': valid_patterns[1:]}
    df = pd.DataFrame(data)

    unique_patterns = sorted(set(valid_patterns), key=lambda p: tuple(p))
    pattern_to_idx = {pattern: idx for idx, pattern in enumerate(unique_patterns)}

    transition_matrix = np.zeros((num_patterns, num_patterns))
    transition_counts = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        current_pattern = row['Current']
        next_pattern = row['Next']
        transition_counts[current_pattern][next_pattern] += 1
        current_idx = pattern_to_idx[current_pattern]
        next_idx = pattern_to_idx[next_pattern]
        transition_matrix[current_idx][next_idx] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

    conditional_probabilities = {current: {next_: count / sum(next_patterns.values())
                                           for next_, count in next_patterns.items()}
                                 for current, next_patterns in transition_counts.items()}

    return conditional_probabilities, transition_matrix, unique_patterns

# %%
# Step 7: Plot the transition matrix heatmap with red-blue colors
def plot_transition_matrix_heatmap(transition_matrix, unique_patterns, D):
    """
    Plots a heatmap of the transition matrix with a red-blue color scheme,
    adjusting figure size and font size dynamically based on D.

    Args:
        transition_matrix (np.ndarray): Transition matrix.
        unique_patterns (list): List of unique ordinal patterns.
    """
    # Plot the heatmap with custom colors and annotations
    plt.imshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)  

    # Add colorbar 
    plt.colorbar() 

    # Set axis labels and title with dynamic font size
    plt.xlabel('Next Pattern')
    plt.ylabel('Current Pattern')
    plt.title(f'Transition Matrix Heatmap for D={D}')


    # Disable tick marks and labels
    plt.xticks([], [])
    plt.yticks([], [])

    plt.tight_layout()
    plt.savefig(f'Transition Matrix Heatmap for D={D}', dpi=300)
    plt.show()

# %%
# Main execution
file_path = r"D:\My physics project\Econophysics\Data\Data_Ex1\Gold\Gold_Processed_Prices.txt"
prices = read_file(file_path)
prices = preprocess_gold_data(prices, window_size=17)

price_diff, normalized_diff = calculate_normalized_price_diff(prices)
plot_y_N(normalized_diff)

calculate_pdf_with_freedman_diaconis(normalized_diff)
for D in range(2,8):
    valid_patterns, pattern_counts = ordinal_pattern(prices, D)

    entropy = calculate_permutation_entropy(pattern_counts, D)
    print(f"Permutation Entropy (D={D}): {entropy}")

    conditional_probabilities, transition_matrix, unique_patterns = calculate_conditional_probabilities_and_matrix(valid_patterns, D)
    for current, probs in conditional_probabilities.items():
        print(f'Conditional probabilities for {current}:')
        for next_pattern, prob in probs.items():
            print(f'  P({next_pattern}|{current}) = {prob:.4f}')
    try:
        plot_transition_matrix_heatmap(transition_matrix, unique_patterns, D)
    except Exception as e:  # Catch the exception and assign it to variable 'e'
        print(f"An error occurred: {e}")  # Print the error message



