import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import os

# style
plt.style.use('seaborn-v0_8-whitegrid')

sizes = [10, 100, 1000]

# distributions
distributions = [
    {
        "name": "Нормальное распределение N(0, 1)",
        "rv": stats.norm(loc=0, scale=1),
        "type": "continuous",
        "xlim": (-4, 4),
        "filename": "1_Normal"
    },
    {
        "name": "Распределение Коши C(0, 1)",
        "rv": stats.cauchy(loc=0, scale=1),
        "type": "continuous",
        "xlim": (-10, 10),
        "filename": "2_Cauchy"
    },
    {
        "name": "Распределение Лапласа L(0, 1/√2)",
        "rv": stats.laplace(loc=0, scale=1/math.sqrt(2)),
        "type": "continuous",
        "xlim": (-4, 4),
        "filename": "3_Laplace"
    },
    {
        "name": "Распределение Пуассона P(10)",
        "rv": stats.poisson(mu=10),
        "type": "discrete",
        "xlim": (0, 24),
        "filename": "4_Poisson"
    },
    {
        "name": "Равномерное распределение U(-√3, √3)",
        "rv": stats.uniform(loc=-math.sqrt(3), scale=2*math.sqrt(3)),
        "type": "continuous",
        "xlim": (-2.5, 2.5),
        "filename": "5_Uniform"
    }
]

# all param for chart
FIG_SIZE = (18, 5)       # size image
HIST_COLOR = '#87CEEB'   # color column (SkyBlue)
EDGE_COLOR = 'black'     # color border
LINE_COLOR = '#DC143C'   # color theoretical line (Crimson)
LINE_WIDTH = 2.5         # size line

for dist in distributions:
    rv = dist["rv"]
    
    # create new figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    fig.suptitle(dist['name'], fontsize=18, fontweight='bold', y=1.05)
    
    for j, n in enumerate(sizes):
        ax = axes[j]
        
        # generation
        data = rv.rvs(size=n)
        
        # axis preparation
        x_min, x_max = dist["xlim"]
        
        # continuous distribution
        if dist["type"] == "continuous":
            # theoretical line
            x = np.linspace(x_min, x_max, 1000)
            y = rv.pdf(x)
            
            # Histogram
            ax.hist(data, bins='auto', density=True, range=(x_min, x_max),
                    color=HIST_COLOR, edgecolor=EDGE_COLOR, alpha=0.7, label='Гистограмма')
            # line
            ax.plot(x, y, color=LINE_COLOR, linewidth=LINE_WIDTH, label='Теория')
            
        # discrete distribution
        else:
            # theoretical point
            x = np.arange(x_min, x_max + 1)
            y = rv.pmf(x)
            
            # Histogram
            bins = np.arange(x_min - 0.5, x_max + 1.5, 1)
            ax.hist(data, bins=bins, density=True,
                    color=HIST_COLOR, edgecolor=EDGE_COLOR, alpha=0.7, label='Гистограмма')
            
            # theoretical markers and lines
            ax.plot(x, y, 'o', color=LINE_COLOR, markersize=6, label='Теория')
            ax.vlines(x, 0, y, colors=LINE_COLOR, linewidth=LINE_WIDTH, alpha=0.5)

        # visual charts settings
        ax.set_title(f"Объем выборки n = {n}", fontsize=14, pad=10)
        ax.set_xlabel("Значение x", fontsize=12)
        if j == 0:
            ax.set_ylabel("Плотность / Вероятность", fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if j == 2:
            ax.legend(fontsize=11, loc='upper right', frameon=True, facecolor='white')

    # visual adjustment and save
    plt.tight_layout()
    
    # Save to file
    file_name = f"{dist['filename']}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"Сохранен файл: {file_name}")
    
    plt.show()
