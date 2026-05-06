import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')

#y = a + b*x
A_TRUE = 2.0
B_TRUE = 2.0

#[-1.8, 2.0]
x = np.linspace(-1.8, 2.0, 20)

#generation y
eps = np.random.normal(0, 1, 20)
y_clean = A_TRUE + B_TRUE * x + eps

#y_1 += 10, y_20 -= 10
y_outliers = y_clean.copy()
y_outliers[0] += 10
y_outliers[-1] -= 10


#MNK
def fit_mnk(x_data, y_data):
    b, a = np.polyfit(x_data, y_data, 1)
    return a, b

#MNM
def fit_mnm(x_data, y_data, initial_guess):
    def objective(params):
        a, b = params
        return np.sum(np.abs(y_data - (a + b * x_data)))

    result = minimize(objective, initial_guess, method='Nelder-Mead')
    return result.x[0], result.x[1]

def create_results_table(a_mnk, b_mnk, a_mnm, b_mnm):
    delta_a_mnk = np.abs(A_TRUE - a_mnk)
    rel_a_mnk = (delta_a_mnk / A_TRUE) * 100
    
    delta_b_mnk = np.abs(B_TRUE - b_mnk)
    rel_b_mnk = (delta_b_mnk / B_TRUE) * 100

    delta_a_mnm = np.abs(A_TRUE - a_mnm)
    rel_a_mnm = (delta_a_mnm / A_TRUE) * 100
    
    delta_b_mnm = np.abs(B_TRUE - b_mnm)
    rel_b_mnm = (delta_b_mnm / B_TRUE) * 100
    
    #DataFrame
    data = {
        'Метод': ['МНК', 'МНМ'],
        'a': [round(a_mnk, 3), round(a_mnm, 3)],
        'Δ a': [round(delta_a_mnk, 3), round(delta_a_mnm, 3)],
        'δ a, %': [round(rel_a_mnk, 2), round(rel_a_mnm, 2)],
        'b': [round(b_mnk, 3), round(b_mnm, 3)],
        'Δ b': [round(delta_b_mnk, 3), round(delta_b_mnm, 3)],
        'δ b, %': [round(rel_b_mnk, 2), round(rel_b_mnm, 2)]
    }
    return pd.DataFrame(data).set_index('Метод')


a_mnk_clean, b_mnk_clean = fit_mnk(x, y_clean)
a_mnm_clean, b_mnm_clean = fit_mnm(x, y_clean, initial_guess=[a_mnk_clean, b_mnk_clean])

a_mnk_out, b_mnk_out = fit_mnk(x, y_outliers)
a_mnm_out, b_mnm_out = fit_mnm(x, y_outliers, initial_guess=[a_mnk_out, b_mnk_out])

print("=================== ВЫБОРКА БЕЗ ВЫБРОСОВ ===================")
print(create_results_table(a_mnk_clean, b_mnk_clean, a_mnm_clean, b_mnm_clean).to_string())
print("\n")
print("=================== ВЫБОРКА С ВЫБРОСАМИ ===================")
print(create_results_table(a_mnk_out, b_mnk_out, a_mnm_out, b_mnm_out).to_string())
print("\n")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Сравнение методов МНК и МНМ (Линейная регрессия)', fontsize=16)

def plot_regression(ax, x_data, y_data, a_mnk, b_mnk, a_mnm, b_mnm, title):
    ax.scatter(x_data, y_data, color='steelblue', edgecolor='black', alpha=0.8, label='Выборка')

    x_line = np.array([-2, 2.2])
    ax.plot(x_line, A_TRUE + B_TRUE * x_line, 'k-', linewidth=2, label='Истинная зависимость')

    ax.plot(x_line, a_mnk + b_mnk * x_line, 'r--', linewidth=2, label='МНК (L2)')

    ax.plot(x_line, a_mnm + b_mnm * x_line, 'b-.', linewidth=2, label='МНМ (L1)')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlim(-2, 2.2)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

plot_regression(axes[0], x, y_clean, a_mnk_clean, b_mnk_clean, a_mnm_clean, b_mnm_clean, 
                "Выборка без аномалий")

plot_regression(axes[1], x, y_outliers, a_mnk_out, b_mnk_out, a_mnm_out, b_mnm_out, 
                "Выборка с выбросами")

plt.tight_layout()
plt.savefig("Lab6_Regression.png", dpi=300)
plt.show()
