import pandas as pd
import numpy as np
from os import path
import seaborn as sns
import matplotlib.pyplot as plt

aspects = ['Service', 'Spatial']
dimension = ['Magnitude', 'Peak', 'Time to Peak', 'Time to Recovery']

def calc_res(df_1, df_2, aspect, dimension, threshold):
    df_1 = df_1[(df_1['Aspect'] == aspect) & (df_1['Dimension'] == dimension) & (df_1['Threshold'] == threshold)]
    df_2 = df_2[(df_2['Aspect'] == aspect) & (df_2['Dimension'] == dimension) & (df_2['Threshold'] == threshold)]
    df = df_1.append(df_2, ignore_index=True)
    return df['Reliability'].mean()

data = []
for n in range(1, 70):
    variant_1 = f'results/Ctown_{n}_reliability.csv'
    variant_2 = f'results_2/Ctown_{n}_reliability.csv'
    if path.exists(variant_1) and path.exists(variant_2):
        df_1 = pd.read_csv(variant_1)
        df_2 = pd.read_csv(variant_2)
        for a in aspects:
            for d in dimension:

                data.append([f'variant_{n}', 'Variant', a, d, calc_res(df_1, df_2, a, d, 0)])



df = pd.DataFrame(data, columns=['Name', 'Type', 'Aspect', 'Dimension', 'Resilience'])


#%%
def fetch_data(reliability, aspect, dimension, threshold=0):

    df = reliability[reliability['Threshold'] == threshold]
    df = df[df['Aspect'] == aspect]
    df = df[df['Dimension'] == dimension]
    new_df = []
    for samples in [500]:
        for r in range(70):
            new_df.append([f'Sample_{r}', 'Benchmark', aspect, dimension, df.sample(samples)['Reliability'].mean()])
    new_df = pd.DataFrame(new_df, columns=['Name', 'Type', 'Aspect', 'Dimension', 'Resilience'])
    return new_df

benchmark = 'Ctown_reliability.csv'
benchmark = pd.read_csv(benchmark)
data = pd.DataFrame()
for a in aspects:
    for d in dimension:
        data = data.append(fetch_data(benchmark, a, d, 1), ignore_index=True)


data = data.append(df, ignore_index=True)
g = sns.FacetGrid(data, row="Dimension", col='Aspect', hue='Type', sharex=False, sharey=False, aspect=2)
g.map(sns.histplot, 'Resilience')
g.add_legend()
plt.show()

