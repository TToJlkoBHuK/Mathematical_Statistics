import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import math

np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')

sizes = [20, 100]
repeats = 1000

distributions = {
    "Нормальное": stats.norm(loc=0, scale=1),
    "Коши": stats.cauchy(loc=0, scale=1),
    "Лапласа": stats.laplace(loc=0, scale=1/math.sqrt(2)),
    "Пуассона": stats.poisson(mu=10),
    "Равномерное": stats.uniform(loc=-math.sqrt(3), scale=2*math.sqrt(3))
}

print("Часть 1. Построение боксплотов...")

colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']

for n in sizes:
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = []
    labels = []
    
    for name, rv in distributions.items():
        data_to_plot.append(rv.rvs(size=n))
        labels.append(name)

    bplot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, whis=1.5,
                       medianprops=dict(color="black", alpha=0.6),
                       flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='gray'))

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('gray')

    ax.set_title(f"Боксплоты Тьюки (объем выборки n = {n})", fontsize=14, pad=15)
    ax.set_xlabel("Распределение", fontsize=12)
    ax.set_ylabel("Значение", fontsize=12)
    ax.grid(axis='y', linestyle='-', alpha=0.4)
    ax.grid(axis='x', visible=False)
    
    ax.set_ylim(-6, 18)

    plt.tight_layout()
    filename = f"Boxplot_n{n}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
print("Графики сохранены (Boxplot_n20.png и Boxplot_n100.png).\n")

print("Часть 2. Экспериментальный расчет доли выбросов (1000 повторений)...")

results_df = pd.DataFrame(index=distributions.keys(), columns=[f"Доля выбросов при n={n}" for n in sizes])

for name, rv in distributions.items():
    for n in sizes:
        samples = rv.rvs(size=(repeats, n))

        Q1 = np.percentile(samples, 25, axis=1, keepdims=True)
        Q3 = np.percentile(samples, 75, axis=1, keepdims=True)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (samples < lower_bound) | (samples > upper_bound)

        outliers_fraction = np.mean(outliers)
        
        results_df.loc[name, f"Доля выбросов при n={n}"] = round(outliers_fraction, 3)

print(f"\n{'='*55}")
print("Средняя доля выбросов по методу Тьюки")
print(f"{'='*55}")
print(results_df.to_string())
print(f"{'='*55}")
