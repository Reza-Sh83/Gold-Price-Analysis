# 📈 Gold Price Time Series Analysis with Entropy and Complexity Measures

This project analyzes the dynamics of gold prices using methods from time series analysis, information theory, and symbolic dynamics. Inspired by approaches in Econophysics, the code investigates the underlying stochastic and nonlinear properties of financial time series.

---

## 🧩 Overview

The workflow includes:

- **Data Preprocessing**: Reading gold price data and applying moving averages to reduce microstructure noise.
- **Statistical Physics Measures**: Computing kernel-based metrics (M¹ and M²) to assess time series complexity.
- **Probability Distribution Function (PDF)**: Constructing the empirical PDF and analyzing the tail with a power-law fit.
- **Ordinal Pattern Analysis**: Embedding time series into ordinal patterns and calculating permutation entropy.
- **Symbolic Dynamics**: Constructing and visualizing the transition matrix between ordinal patterns.

---

## 🛠 Features

### ✅ Data Preprocessing
- Reads price data from CSV and computes log-returns.
- Applies moving average smoothing to reduce noise.

### 🔍 Complexity Metrics
- Computes **M¹** and **M²** kernel-based metrics:
  - M¹ = \(\sqrt{\langle (X_{t+1} - X_t)^2 \rangle}\)
  - M² = \(\langle |X_{t+1} - X_t| \rangle\)

### 📊 PDF and Tail Fitting
- Uses the Freedman–Diaconis rule to create a histogram.
- Analyzes tail behavior by fitting \( f(y) \sim \frac{1}{y^b} \) for large \( y \).

### 🔢 Ordinal Pattern Analysis
- Embeds time series into ordinal patterns of order \( D \).
- Counts pattern frequencies and computes **normalized permutation entropy**.

### 🔁 Transition Matrix
- Constructs a matrix showing transition probabilities between ordinal patterns.
- Visualizes with heatmaps.

---

## 🧠 Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy

Install dependencies with:

```bash
pip install numpy pandas matplotlib scipy
```

---

## 📌 How to Use

1. Place your gold price CSV file as `gold_prices.csv` in the root directory.
2. Run the analysis script:

```bash
python gold_analysis.py
```

3. Output will include:
   - Kernel-based measures
   - PDF histogram and power-law tail fit
   - Permutation entropy vs embedding dimension
   - Transition matrices for symbolic dynamics

---

## 📚 References

- Bandt, C., & Pompe, B. (2002). *Permutation entropy: A natural complexity measure for time series*. Physical review letters.
- Mantegna, R. N., & Stanley, H. E. (1999). *An introduction to Econophysics: Correlations and complexity in finance*.
- Zunino, L., et al. (2008). *Characterization of chaotic maps using permutation entropy*.

---

## © License

MIT License. Feel free to modify and use for academic or personal purposes. Attribution appreciated!
