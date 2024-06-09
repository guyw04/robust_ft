import numpy as np
import matplotlib.pyplot as plt

# Create two non-Gaussian, fairly random distributions A and B
np.random.seed(42)  # For reproducibility

# Example distributions (uniform and triangular for simplicity)
A = np.random.uniform(-3, 3, 1000)
B = np.random.triangular(-2, 0, 2, 1000)

# Compute key statistics
mu_A, sigma_A = np.mean(A), np.std(A)
mu_B, sigma_B = np.mean(B), np.std(B)

# Initialize gamma
gamma = 1.  # Starting with a balanced influence

# Define the projection operation
def project(A, gamma, mu_A, sigma_A, mu_B, sigma_B):
    A_proj = gamma * ((A - mu_A) / sigma_A) * sigma_B + mu_B + (1 - gamma) * A
    return A_proj

# Project A towards B
A_proj = project(A, gamma, mu_A, sigma_A, mu_B, sigma_B)

# Visualization
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.hist(A, bins=30, alpha=0.7, label='A')
plt.hist(B, bins=30, alpha=0.7, label='B')
plt.title("Original Distributions")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(A_proj, bins=30, alpha=0.7, label='A_proj')
plt.hist(B, bins=30, alpha=0.7, label='B')
plt.title("Projected Distribution")
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(A, bins=30, alpha=0.7, label='A')
plt.hist(A_proj, bins=30, alpha=0.7, label='A_proj')
plt.title("A vs. A_proj")
plt.legend()

plt.tight_layout()
plt.show()
