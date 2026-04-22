import numpy as np
import scipy.stats as stats
import pandas as pd
import math

np.random.seed(42)

sizes = [10, 100, 1000]
repeats = 1000

distributions = {
    "Нормальное N(0,1)": stats.norm(loc=0, scale=1),
    "Коши C(0,1)": stats.cauchy(loc=0, scale=1),
    "Лапласа L(0,1/√2)": stats.laplace(loc=0, scale=1/math.sqrt(2)),
    "Пуассона P(10)": stats.poisson(mu=10),
    "Равномерное U(-√3, √3)": stats.uniform(loc=-math.sqrt(3), scale=2*math.sqrt(3))
}

def get_E_and_D(z_array):
    E_z = np.mean(z_array)
    D_z = np.mean(z_array**2) - E_z**2 
    return E_z, D_z

def format_estimate(E, D):
    if D < 0: D = 0
        
    SD = math.sqrt(D)
    
    if SD == 0:
        return f"{E:.2f} ± 0.00"
    order = math.floor(math.log10(SD))

    decimals = -order
    if decimals < 0: 
        decimals = 0
    SD_rounded = round(SD, decimals)
    E_rounded = round(E, decimals)

    if decimals > 0:
        format_str = f"{{:.{decimals}f}} ± {{:.{decimals}f}}"
    else:
        format_str = "{:.0f} ± {:.0f}"
        
    return format_str.format(E_rounded, SD_rounded)

for dist_name, rv in distributions.items():
    print(f"\n{'='*65}")
    print(f"Распределение: {dist_name}")
    print(f"{'='*65}")
    
    #DataFrame
    df_results = pd.DataFrame(index=["Среднее (mean)", "Медиана (med)", 
                                     "Полусумма экстр. (Z_R)", "Полусумма кварт. (Z_Q)", 
                                     "Усеченное среднее (Z_tr)"])
    
    for n in sizes:
        #matrix 1000 * n
        samples = rv.rvs(size=(repeats, n))

        z_mean = np.mean(samples, axis=1)
        z_med = np.median(samples, axis=1)
        z_r = (np.min(samples, axis=1) + np.max(samples, axis=1)) / 2
        z_q = (np.percentile(samples, 25, axis=1) + np.percentile(samples, 75, axis=1)) / 2
        z_tr = stats.trim_mean(samples, proportiontocut=0.1, axis=1)

        characteristics = [z_mean, z_med, z_r, z_q, z_tr]

        results_str = []
        for z_array in characteristics:
            E_z, D_z = get_E_and_D(z_array)
            results_str.append(format_estimate(E_z, D_z))

        df_results[f'Оценка (n={n})'] = results_str

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results.to_string())
    print("\n")
