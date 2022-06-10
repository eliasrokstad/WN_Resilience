import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np
sns.set_style("white")
sns.set_palette("Spectral")

def plot_resilience(reliability, aspect, dimension='Magnitude', threshold=1):
    df = reliability[reliability['Threshold'] == threshold]
    df = df[df['Aspect'] == aspect]
    df = df[df['Dimension'] == dimension]

    sns.scatterplot(x=df['Stress'], y=df['Reliability'])
    sns.lineplot(x=df['Stress'], y=df['Reliability'], alpha=0.5)
    plt.yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    plt.xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    plt.title(f'{dimension} of {aspect} Dimension, Threshold = {threshold} ')
    plt.show()
    sns.boxplot(data=result, x='Aspect', y='Reliaility', hue='Dimension')
    plt.show()

def resilience(reliability, aspect='Spatial', dimension='Magnitude', threshold=1):
    df = reliability[reliability['Threshold'] == threshold]
    df = df[df['Simulation'] != 0]
    result = df.groupby(['Simulation', 'Aspect', 'Dimension'], as_index=False).mean()
    result = result[['Simulation', 'Aspect', 'Dimension', 'Reliability']]
    sns.boxplot(data=result, x='Dimension', y='Reliability', hue='Aspect')
    plt.title('Variability - Summary')
    plt.show()

    # Spatial magnitude show the highest variability
    return result

def stability(reliability, aspect, dimension, threshold):
    df = reliability[reliability['Threshold'] == threshold]
    df = df[df['Aspect'] == aspect]
    df = df[df['Dimension'] == dimension]
    df = df[df['Simulation'] != 0]
    if df.size == 0:
        return [200, aspect, dimension,threshold, np.NaN, np.NaN]
    result = df.groupby(['Simulation'], as_index=False).mean()
    data = []
    for samples in [200]:
        for r in range(1000):
            data.append([samples, result.sample(samples)['Reliability'].mean()])
    data = pd.DataFrame(data, columns=['N samples', 'Resilience'])
    sns.boxplot(x=data['N samples'], y=data['Resilience'])
    plt.title(f'{aspect} - {dimension}')
    plt.show()

    for n in [200]:
        a = data[data['N samples'] == n]['Resilience'].to_numpy()
        ci = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
        ci = (ci[1] - ci[0])/2
        print(f'{n} samples: ', a.mean(), '+-', ci)

    return [200, aspect, dimension,threshold, a.mean(), ci]

def calc_res(data, aspect, dimension, threshold):
    df = data[data['Aspect'] == aspect]
    df = df[df['Dimension'] == dimension]
    df = df[df['Threshold'] == threshold]
    return df['Reliability'].mean()

def fetch_data(reliability, aspect, dimension, threshold=0):
    df = reliability[reliability['Threshold'] == threshold]
    df = df[df['Aspect'] == aspect]
    df = df[df['Dimension'] == dimension]
    result = []
    for samples in [5, 10, 25, 50, 100, 200, 400]:
        temp = []
        for r in range(100):
            temp.append([df.sample(samples)['Reliability'].mean()])

        a = np.array(temp)
        ci = st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))


        result.append([samples, f'{aspect} of {dimension} resilience', 'mean', np.mean(a)])
        result.append([samples, f'{aspect} of {dimension} resilience', 'upper', ci[0][0]])
        result.append([samples, f'{aspect} of {dimension} resilience', 'lower', ci[1][0]])
    return pd.DataFrame(result, columns=['Samples', 'Dimension', 'Tag', 'Value'])



data = pd.read_csv('Ctown_reliability.csv', index_col=0)

aspects = ['Service', 'Spatial']
dimension = ['Magnitude', 'Peak']
df = pd.DataFrame()
for a in aspects:
    for d in dimension:
        df = df.append(fetch_data(data, a, d), ignore_index=True)


y_vars, x_vars = 'Value', 'Samples'

g = sns.FacetGrid(df, col='Dimension', sharey=False)
g.map(sns.lineplot, x_vars, y_vars)
g.set_xticks([10, 20, 50, 100, 200, 300])
g.set_xticklabels([10, 20, 50, 100, 200, 300])
plt.show()




