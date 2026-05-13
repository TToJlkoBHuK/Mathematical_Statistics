import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')

#hi^2
def chi_square_normality_test(data, alpha=0.05):
    n = len(data)

    mu_hat = np.mean(data)
    sigma_hat = np.std(data, ddof=0)
    
    #n/k >= 5  =>  k <= n/5
    if n == 100:
        k = 10  #100/10 = 10 (>= 5)
    elif n == 20:
        k = 4   #20/4 = 5 (>= 5)
    else:
        k = int(n / 5)
        
    expected_freq = n / k

    # P(X < x_i) = i / k
    edges = [stats.norm.ppf(i/k, loc=mu_hat, scale=sigma_hat) for i in range(1, k)]
    edges_full = [-np.inf] + edges + [np.inf]

    observed_freq, _ = np.histogram(data, bins=edges_full)

    #((O_i - E_i)^2 / E_i)
    chi2_calc = np.sum((observed_freq - expected_freq)**2 / expected_freq)
    
    #df = k - 1 - p
    df = k - 1 - 2
    chi2_crit = stats.chi2.ppf(1 - alpha, df)

    p_value = 1 - stats.chi2.cdf(chi2_calc, df)

    reject_h0 = chi2_calc > chi2_crit
    
    return {
        'n': n,
        'Оценка μ (ММП)': round(mu_hat, 4),
        'Оценка σ (ММП)': round(sigma_hat, 4),
        'Число интервалов (k)': k,
        'Степени свободы (df)': df,
        'χ² (расчетный)': round(chi2_calc, 4),
        'χ² (критический)': round(chi2_crit, 4),
        'p-value': round(p_value, 4),
        'Отвергаем H0?': 'ДА' if reject_h0 else 'НЕТ'
    }, edges, mu_hat, sigma_hat


#generation + tests
datasets = {
    'Нормальное N(0, 1)': stats.norm(loc=0, scale=1).rvs(100),
    'Равномерное U(-√3, √3)': stats.uniform(loc=-math.sqrt(3), scale=2*math.sqrt(3)).rvs(20),
    'Лапласа L(0, 1/√2)': stats.laplace(loc=0, scale=1/math.sqrt(2)).rvs(20)
}

results = []
plot_data = {}

for name, data in datasets.items():
    res, edges, mu, sig = chi_square_normality_test(data)
    res_with_name = {'Распределение (Истинное)': name}
    res_with_name.update(res)
    results.append(res_with_name)
    plot_data[name] = (data, edges, mu, sig)

df_results = pd.DataFrame(results).set_index('Распределение (Истинное)')
print("========================== РЕЗУЛЬТАТЫ ПРОВЕРКИ ГИПОТЕЗ (КРИТЕРИЙ ПИРСОНА) ==========================")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_results.to_string())
print("====================================================================================================")


#visual
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Проверка гипотезы о нормальности (Разбиение на равновероятные интервалы)', fontsize=16, y=1.05)

for i, (name, (data, edges, mu, sig)) in enumerate(plot_data.items()):
    ax = axes[i]
    n = len(data)
    
    plot_edges = [np.min(data)-0.5] + edges + [np.max(data)+0.5]
    
    ax.hist(data, bins=plot_edges, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Гистограмма')

    x_lin = np.linspace(plot_edges[0], plot_edges[-1], 500)
    ax.plot(x_lin, stats.norm.pdf(x_lin, loc=mu, scale=sig), 'r-', lw=2, label=f'ММП Оценка N({mu:.2f}, {sig:.2f}²)')

    for edge in edges:
        ax.axvline(edge, color='green', linestyle='--', alpha=0.6)
        
    ax.set_title(f"{name}\n(n={n})", fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    
plt.tight_layout()
plt.savefig("Lab7_ChiSquare.png", dpi=300, bbox_inches='tight')
plt.show()
