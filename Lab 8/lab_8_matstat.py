import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

alpha = 0.05

#generation
n1, n2 = 20, 100
sample1 = stats.norm(loc=0, scale=1).rvs(n1)
sample2 = stats.norm(loc=0, scale=1).rvs(n2)

def calculate_confidence_intervals(data, alpha):
    n = len(data)
    df = n - 1

    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1) 
    std_val = np.sqrt(var_val)
    
    #t-распределение Стьюдента
    t_crit = stats.t.ppf(1 - alpha/2, df)
    margin_mean = t_crit * (std_val / np.sqrt(n))
    ci_mean = (mean_val - margin_mean, mean_val + margin_mean)
    
    #Хи-квадрат распределение
    chi2_lower_crit = stats.chi2.ppf(alpha/2, df)
    chi2_upper_crit = stats.chi2.ppf(1 - alpha/2, df)
    
    ci_var_lower = (df * var_val) / chi2_upper_crit
    ci_var_upper = (df * var_val) / chi2_lower_crit
    ci_var = (ci_var_lower, ci_var_upper)
    
    return {
        'Объем (n)': n,
        'Точечная оценка μ (x̄)': mean_val,
        'ДИ для μ (нижняя)': ci_mean[0],
        'ДИ для μ (верхняя)': ci_mean[1],
        'Точечная оценка σ² (s²)': var_val,
        'ДИ для σ² (нижняя)': ci_var[0],
        'ДИ для σ² (верхняя)': ci_var[1]
    }

res_20 = calculate_confidence_intervals(sample1, alpha)
res_100 = calculate_confidence_intervals(sample2, alpha)

df_ci = pd.DataFrame([res_20, res_100]).round(4)
df_ci.insert(0, 'Выборка', ['Выборка 1', 'Выборка 2'])
df_ci = df_ci.set_index('Выборка')

print("=========================== ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (Уровень доверия 95%) ===========================")
print(df_ci.to_string())
print("=====================================================================================================\n")


#F-тест
#H0: σ1² = σ2²
#H1: σ1² ≠ σ2²
def f_test_equal_variances(data1, data2, alpha):
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)
    df1 = len(data1) - 1
    df2 = len(data2) - 1
    
    #F-статистика
    f_stat = var1 / var2
    
    #p-value
    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

    f_crit_lower = stats.f.ppf(alpha/2, df1, df2)
    f_crit_upper = stats.f.ppf(1 - alpha/2, df1, df2)
    
    reject_h0 = p_value < alpha
    
    return f_stat, f_crit_lower, f_crit_upper, p_value, reject_h0

f_stat, f_low, f_up, p_val, reject = f_test_equal_variances(sample1, sample2, alpha)

print("========================= F-ТЕСТ О РАВЕНСТВЕ ДИСПЕРСИЙ (Критерий Фишера) =========================")
print(f"H0: Дисперсии равны (σ1² = σ2²)")
print(f"H1: Дисперсии не равны (σ1² ≠ σ2²)")
print(f"Отношение дисперсий s1² / s2² (F-наблюдаемое) = {f_stat:.4f}")
print(f"Критические границы F (нижн, верхн)         = ({f_low:.4f}, {f_up:.4f})")
print(f"P-value                                     = {p_val:.4f}")
print(f"Отвергаем H0?                               = {'ДА' if reject else 'НЕТ'}")
print("=====================================================================================================\n")


#visual
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Сравнение точечных и интервальных оценок (n=20 vs n=100)', fontsize=14)

y_pos = [1, 2]
labels = [f'Выборка 1 (n={n1})', f'Выборка 2 (n={n2})']

means = [res_20['Точечная оценка μ (x̄)'], res_100['Точечная оценка μ (x̄)']]
mean_err_lower = [means[0] - res_20['ДИ для μ (нижняя)'], means[1] - res_100['ДИ для μ (нижняя)']]
mean_err_upper = [res_20['ДИ для μ (верхняя)'] - means[0], res_100['ДИ для μ (верхняя)'] - means[1]]

ax1.errorbar(means, y_pos, xerr=[mean_err_lower, mean_err_upper], fmt='o', color='b', 
             capsize=8, capthick=2, markersize=8, label='Оценка и 95% ДИ')
ax1.axvline(0, color='r', linestyle='--', alpha=0.5, label='Истинное μ=0')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels)
ax1.set_title('Доверительные интервалы для мат. ожидания (μ)')
ax1.legend()

vars = [res_20['Точечная оценка σ² (s²)'], res_100['Точечная оценка σ² (s²)']]
var_err_lower = [vars[0] - res_20['ДИ для σ² (нижняя)'], vars[1] - res_100['ДИ для σ² (нижняя)']]
var_err_upper = [res_20['ДИ для σ² (верхняя)'] - vars[0], res_100['ДИ для σ² (верхняя)'] - vars[1]]

ax2.errorbar(vars, y_pos, xerr=[var_err_lower, var_err_upper], fmt='s', color='g', 
             capsize=8, capthick=2, markersize=8, label='Оценка и 95% ДИ')
ax2.axvline(1, color='r', linestyle='--', alpha=0.5, label='Истинная σ²=1')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.set_title('Доверительные интервалы для дисперсии (σ²)')
ax2.legend()

plt.tight_layout()
plt.savefig("Lab8_ConfidenceIntervals.png", dpi=300, bbox_inches='tight')
plt.show()
