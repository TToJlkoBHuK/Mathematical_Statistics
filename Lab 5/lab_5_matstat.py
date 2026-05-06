import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')

sizes = [20, 60, 100]
repeats = 1000
scenarios = ['p=0', 'p=0.5', 'p=0.9', 'mixture']

#generation 2D
def get_normal_2d(n, rho):
    cov_matrix = [[1, rho], [rho, 1]]
    return np.random.multivariate_normal([0, 0], cov_matrix, n)

#generation mixture
def get_mixture(n):
    #90%
    cov1 = [[1, 0.9], [0.9, 1]]
    #10%
    cov2 = [[100, -90], [-90, 100]] 

    choices = np.random.rand(n)
    mask1 = choices < 0.9
    mask2 = ~mask1
    
    data = np.zeros((n, 2))
    if np.sum(mask1) > 0:
        data[mask1] = np.random.multivariate_normal([0, 0], cov1, np.sum(mask1))
    if np.sum(mask2) > 0:
        data[mask2] = np.random.multivariate_normal([0, 0], cov2, np.sum(mask2))
    return data

#correlation
def quadrant_correlation(x, y):
    med_x, med_y = np.median(x), np.median(y)
    return np.mean(np.sign(x - med_x) * np.sign(y - med_y))

#ellips
def plot_ellipse(ax, x, y):
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    c = 2.447
    width, height = 2 * c * np.sqrt(np.abs(eigvals))

    ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle,
                      edgecolor='red', facecolor='none', linestyle='--', linewidth=1.2)
    ax.add_patch(ellipse)


print("Часть 1. Вычисление коэффициентов (Монте-Карло, 1000 повторений)...\n")

for n in sizes:
    #DataFram
    columns = ['Pearson μ', 'Pearson σ²', 'Spearman μ', 'Spearman σ²', 'Quadrant μ', 'Quadrant σ²']
    df = pd.DataFrame(index=scenarios, columns=columns)
    
    for scenario in scenarios:
        pearson_list, spearman_list, quadrant_list = [], [], []
        
        for _ in range(repeats):
            if scenario == 'p=0':
                data = get_normal_2d(n, 0)
            elif scenario == 'p=0.5':
                data = get_normal_2d(n, 0.5)
            elif scenario == 'p=0.9':
                data = get_normal_2d(n, 0.9)
            elif scenario == 'mixture':
                data = get_mixture(n)
                
            x, y = data[:, 0], data[:, 1]
            
            #Pearson
            p_corr, _ = stats.pearsonr(x, y)
            pearson_list.append(p_corr)
            
            #Spearman
            s_corr, _ = stats.spearmanr(x, y)
            spearman_list.append(s_corr)
            
            #Quadrant
            q_corr = quadrant_correlation(x, y)
            quadrant_list.append(q_corr)

        df.loc[scenario, 'Pearson μ'] = f"{np.mean(pearson_list):.6f}"
        df.loc[scenario, 'Pearson σ²'] = f"{np.var(pearson_list):.6f}"
        df.loc[scenario, 'Spearman μ'] = f"{np.mean(spearman_list):.6f}"
        df.loc[scenario, 'Spearman σ²'] = f"{np.var(spearman_list):.6f}"
        df.loc[scenario, 'Quadrant μ'] = f"{np.mean(quadrant_list):.6f}"
        df.loc[scenario, 'Quadrant σ²'] = f"{np.var(quadrant_list):.6f}"
        
    print(f"========================= Таблица для n = {n} =========================")
    print(df.to_string())
    print("\n")


print("Часть 2. Генерация диаграмм рассеяния и эллипсов...\n")

titles = {
    'p=0': 'Сравнение для ρ=0',
    'p=0.5': 'Сравнение для ρ=0.5',
    'p=0.9': 'Сравнение для ρ=0.9',
    'mixture': 'Сравнение для смеси (mixture)'
}

for scenario in scenarios:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(titles[scenario], fontsize=16, y=1.05)
    
    for i, n in enumerate(sizes):
        ax = axes[i]

        if scenario == 'p=0':
            data = get_normal_2d(n, 0)
        elif scenario == 'p=0.5':
            data = get_normal_2d(n, 0.5)
        elif scenario == 'p=0.9':
            data = get_normal_2d(n, 0.9)
        elif scenario == 'mixture':
            data = get_mixture(n)
            
        x, y = data[:, 0], data[:, 1]
        
        #Scatter plot
        ax.scatter(x, y, alpha=0.6, s=30, edgecolor='white', linewidth=0.5, color='steelblue')

        plot_ellipse(ax, x, y)

        ax.set_title(f"n={n}", fontsize=14)
        ax.grid(True, linestyle='-', alpha=0.3)

        ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    # Сохраняем график
    filename = f"Lab5_{scenario.replace('=', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("Все 4 графика успешно сгенерированы и сохранены!")
