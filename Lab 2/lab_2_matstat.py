import numpy as np
import scipy.stats as stats
import pandas as pd
import math

# ifx seed
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

#E(z) D(z)
def get_E_and_D(z_array):
    E_z = np.mean(z_array)
    #D(z) = mean(z^2) - (mean(z))^2
    D_z = np.mean(z_array**2) - E_z**2 
    return E_z, D_z

for dist_name, rv in distributions.items():
    print(f"\n{'='*70}")
    print(f"Распределение: {dist_name}")
    print(f"{'='*70}")
    
    #create DataFrame
    df_results = pd.DataFrame(index=["Среднее (mean)", "Медиана (med)", "Полусумма экстр. (Z_R)", "Полусумма кварт. (Z_Q)", "Усеченное среднее (Z_tr)"])
    
    for n in sizes:
        # matrix = 1000 * n
        samples = rv.rvs(size=(repeats, n))
        
        #Выборочное среднее
        z_mean = np.mean(samples, axis=1)
        
        #Медиана
        z_med = np.median(samples, axis=1)
        
        #Полусумма экстремальных элементов
        z_r = (np.min(samples, axis=1) + np.max(samples, axis=1)) / 2
        
        #Полусумма квартилей
        z_q = (np.percentile(samples, 25, axis=1) + np.percentile(samples, 75, axis=1)) / 2
        
        #Усеченное среднее
        z_tr = stats.trim_mean(samples, proportiontocut=0.1, axis=1)

        characteristics = [z_mean, z_med, z_r, z_q, z_tr]
        
        E_values = []
        D_values = []
        
        for z_array in characteristics:
            E_z, D_z = get_E_and_D(z_array)
            #0.000001
            E_values.append(round(E_z, 6))
            D_values.append(round(D_z, 6))

        df_results[f'E(z), n={n}'] = E_values
        df_results[f'D(z), n={n}'] = D_values

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results.to_string())
    print("\n")
