# 📈 Econophysics: Gold Price Analysis

This project applies **Econophysics techniques** to analyze gold price data. It includes advanced statistical and nonlinear time-series analysis, such as permutation entropy and transition matrices. The goal is to understand the complex dynamics of gold prices using tools inspired by physics.

---

## 🚀 Features

- **Noise reduction** using moving averages
- **PDF estimation** with Freedman-Diaconis binning
- **Power-law fit** to tails of distribution
- **Ordinal pattern analysis**
- **Permutation entropy** (normalized)
- **Transition matrix and conditional probability** heatmaps

---

## 🧪 Methodology

1. **Preprocessing**: Smooth price data to remove microstructure noise
2. **Normalized differences**: Standardized returns (detrended)
3. **PDF Plotting**: Histogram (log-log scale) with fitted tail
4. **Ordinal Pattern Detection**:
   - Sliding window of size `D` to create rank-based patterns
   - Filter for unique permutations only
5. **Permutation Entropy**:
   - Entropy of ordinal pattern frequency distribution
   - Normalized by maximum entropy (log of factorial D)
6. **Conditional Probabilities**:
   - Compute transitions between consecutive patterns
   - Visualize as a **heatmap** of transition matrix

---

## 📊 Example Output

### Normalized Price Differences
<img src="Normalized Difference y(N).png" width="450">

### PDF in Log-Log Scale
<img src="Probability Density Function (PDF) - Log-Log Scale.png" width="450">

### Permutation Entropy (Sample Output)
```
Permutation Entropy (D=2): 0.999975
Permutation Entropy (D=3): 0.674493
Permutation Entropy (D=4): 0.551636
...
```

### Transition Matrix Heatmap (for D=5)
<img src="Transition Matrix Heatmap for D=5.png" width="450">

---

## 📌 Requirements

```bash
numpy
matplotlib
scipy
seaborn
pandas
```

---

## ▶️ How to Run

```python
# In Jupyter Notebook or Python script
file_path = "path/to/Gold_Processed_Prices.txt"
prices = read_file(file_path)
prices = preprocess_gold_data(prices)
... # Continue with analysis
```

## 📚 References

- Bandt, C., & Pompe, B. (2002). *Permutation entropy: A natural complexity measure for time series*. Physical review letters.
- Mantegna, R. N., & Stanley, H. E. (1999). *An introduction to Econophysics: Correlations and complexity in finance*.
- Zunino, L., et al. (2008). *Characterization of chaotic maps using permutation entropy*.

---

## © License

MIT License. Feel free to modify and use for academic or personal purposes. Attribution appreciated!
