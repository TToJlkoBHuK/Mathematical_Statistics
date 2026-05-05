import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')

#generation
n = 100
rv = stats.cauchy(loc=0, scale=1)
sample = rv.rvs(size=n)

x_range = np.linspace(-4, 4, 1000)
y_theory = rv.pdf(x_range)

multipliers = [0.1, 1.0, 2.5]
x_labels = [
    "Недосглаживание (зубчатость)", 
    "Стандарт (правило Скотта)", 
    "Пересглаживание (потеря формы)"
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Влияние h на ядерную оценку (Коши, n=100)", fontsize=16, y=1.05)

for i, m in enumerate(multipliers):
    ax = axes[i]
    
    #KDE
    kde = stats.gaussian_kde(sample)

    default_factor = kde.factor
    kde.set_bandwidth(bw_method = default_factor * m)

    y_kde = kde(x_range)

    ax.plot(x_range, y_theory, color='red', linewidth=2, label='Теория')
    ax.plot(x_range, y_kde, color='blue', linewidth=1.5, label='KDE')

    ax.set_title(f"h = {m}h", fontsize=14)
    ax.set_xlabel(x_labels[i], fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(-4, 4)

    ax.set_ylim(0, 0.34)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig("Lab4_Bandwidth_Effect.png", dpi=300, bbox_inches='tight')
plt.show()
