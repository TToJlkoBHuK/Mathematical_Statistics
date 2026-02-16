import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

#style
plt.style.use('seaborn-v0_8-whitegrid')

sizes = [10, 100, 1000]

#distributions
distributions = [
    {
        "name": "Нормальное распределение N(0, 1)",
        "rv": stats.norm(loc=0, scale=1),
        "type": "continuous",
        "xlim": (-4, 4)
    },
    {
        "name": "Распределение Коши C(0, 1)",
        "rv": stats.cauchy(loc=0, scale=1),
        "type": "continuous",
        "xlim": (-10, 10) 
    },
    {
        "name": "Распределение Лапласа L(0, 1/√2)",
        "rv": stats.laplace(loc=0, scale=1/math.sqrt(2)),
        "type": "continuous",
        "xlim": (-4, 4)
    },
    {
        "name": "Распределение Пуассона P(10)",
        "rv": stats.poisson(mu=10),
        "type": "discrete",
        "xlim": (0, 24)
    },
    {
        "name": "Равномерное распределение U(-√3, √3)",
        "rv": stats.uniform(loc=-math.sqrt(3), scale=2*math.sqrt(3)),
        "type": "continuous",
        "xlim": (-2.5, 2.5)
    }
]

#all param for chart
FIG_SIZE = (10, 6)       #size image
HIST_COLOR = '#87CEEB'   #color colomn (SkyBlue)
EDGE_COLOR = 'black'     #color border
LINE_COLOR = '#DC143C'   #color theorytical line (Crimson)
LINE_WIDTH = 3           #size line

for dist in distributions:
    rv = dist["rv"]
    
    for n in sizes:
        #create new figure for chart
        plt.figure(figsize=FIG_SIZE)
        
        #generation
        data = rv.rvs(size=n)
        
        #axis preparation
        x_min, x_max = dist["xlim"]
        
        #continuous distribution
        if dist["type"] == "continuous":
            #theorytical line
            x = np.linspace(x_min, x_max, 1000)
            y = rv.pdf(x)
            
            #Histogram
            # range=(x_min, x_max) cuts off the tails
            plt.hist(data, bins='auto', density=True, range=(x_min, x_max),
                     color=HIST_COLOR, edgecolor=EDGE_COLOR, alpha=0.7, label='Гистограмма выборки')
            #line
            plt.plot(x, y, color=LINE_COLOR, linewidth=LINE_WIDTH, label='Теоретическая плотность')
            
        #discrete distribution
        else:
            #theorytical point
            x = np.arange(x_min, x_max + 1)
            y = rv.pmf(x)
            
            #Histogram (bin alignment by integers)
            bins = np.arange(x_min - 0.5, x_max + 1.5, 1)
            plt.hist(data, bins=bins, density=True,
                     color=HIST_COLOR, edgecolor=EDGE_COLOR, alpha=0.7, label='Гистограмма выборки')
            
            #theorytical markers and lines
            plt.plot(x, y, 'o', color=LINE_COLOR, markersize=8, label='Теоретическая вероятность')
            plt.vlines(x, 0, y, colors=LINE_COLOR, linewidth=LINE_WIDTH, alpha=0.5)

        #visual charts
        plt.title(f"{dist['name']}\nОбъем выборки n = {n}", fontsize=16, pad=15)
        plt.xlabel("Значение x", fontsize=14)
        plt.ylabel("Плотность вероятности", fontsize=14)
        plt.xlim(x_min, x_max)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        
        #visual
        plt.tight_layout()
        plt.show()
