import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')

sizes = [20, 60, 100]

distributions = {
    "Normal $N(0, 1)$": {
        "rv": stats.norm(loc=0, scale=1),
        "type": "continuous",
        "x_range": (-4, 4),
        "filename": "Lab4_1_Normal"
    },
    "Cauchy $C(0, 1)$": {
        "rv": stats.cauchy(loc=0, scale=1),
        "type": "continuous",
        "x_range": (-4, 4),
        "filename": "Lab4_2_Cauchy"
    },
    "Laplace $L(0, 1/\sqrt{2})$": {
        "rv": stats.laplace(loc=0, scale=1/math.sqrt(2)),
        "type": "continuous",
        "x_range": (-4, 4),
        "filename": "Lab4_3_Laplace"
    },
    "Poisson $P(10)$": {
        "rv": stats.poisson(mu=10),
        "type": "discrete",
        "x_range": (6, 14),
        "filename": "Lab4_4_Poisson"
    },
    "Uniform $U(-\sqrt{3}, \sqrt{3})$": {
        "rv": stats.uniform(loc=-math.sqrt(3), scale=2*math.sqrt(3)),
        "type": "continuous",
        "x_range": (-4, 4),
        "filename": "Lab4_5_Uniform"
    }
}

print("Генерация графиков (ЭФР и KDE)...")

for name, data in distributions.items():
    rv = data["rv"]
    x_min, x_max = data["x_range"]
    dist_type = data["type"]
    file_prefix = data["filename"]
    
    #generation CDF/KDE
    samples = {n: rv.rvs(size=n) for n in sizes}

    fig_cdf, axes_cdf = plt.subplots(1, 3, figsize=(18, 5))
    fig_cdf.suptitle(name, fontsize=16, y=1.05)
    
    for i, n in enumerate(sizes):
        ax = axes_cdf[i]
        sample = samples[n]
        
        #theory red line
        x_theory = np.linspace(x_min, x_max, 1000)
        y_theory = rv.cdf(x_theory)
        ax.plot(x_theory, y_theory, color='red', linewidth=2, label='Теория')
        
        #empire blue line
        x_ecdf = np.sort(sample)
        y_ecdf = np.arange(1, n + 1) / n

        x_plot = np.concatenate(([x_min], x_ecdf, [x_max]))
        y_plot = np.concatenate(([0], y_ecdf, [1]))
        
        ax.step(x_plot, y_plot, where='post', color='blue', linewidth=1.5, label='Эмпирическое')

        ax.set_title(f"n = {n}", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left', fontsize=10, frameon=True)
            
    plt.tight_layout()
    fig_cdf.savefig(f"{file_prefix}_CDF.png", dpi=300, bbox_inches='tight')
    plt.close(fig_cdf)

    fig_kde, axes_kde = plt.subplots(1, 3, figsize=(18, 5))
    fig_kde.suptitle(name, fontsize=16, y=1.05)
    
    for i, n in enumerate(sizes):
        ax = axes_kde[i]
        sample = samples[n]
        
        #theory p
        x_theory = np.linspace(x_min, x_max, 1000)
        if dist_type == "continuous":
            y_theory = rv.pdf(x_theory)
            ax.plot(x_theory, y_theory, color='red', linewidth=2, label='Теория')
        else:
            x_int = np.arange(x_min, x_max + 1)
            y_pmf = rv.pmf(x_int)
            ax.plot(x_int, y_pmf, 'ro-', linewidth=1.5, markersize=5, label='Теория')
            
        #yader cost
        kde = stats.gaussian_kde(sample)
        y_kde = kde(x_theory)
        ax.plot(x_theory, y_kde, color='blue', linewidth=1.5, label='Ядерное')

        ax.set_title(f"n = {n}", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=10, frameon=True)
            
    plt.tight_layout()
    fig_kde.savefig(f"{file_prefix}_KDE.png", dpi=300, bbox_inches='tight')
    plt.close(fig_kde)

print("Все 10 графиков успешно сгенерированы и сохранены!")
