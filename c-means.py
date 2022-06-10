from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

def absoluteFilePaths(directory):
    for root, dirs, files in os.walk(os.path.abspath(directory)):
        return [os.path.join(root, file) for file in files]


def get_resilience(directory):
    results_files = absoluteFilePaths(directory)
    df = pd.DataFrame()
    for file in results_files:
        temp_df = pd.read_csv(file, index_col=0).groupby(['Dimension', 'Aspect', 'Threshold']).mean().reset_index()
        temp_df = temp_df[['Dimension', 'Aspect', 'Threshold', 'Reliability']].rename(columns={'Reliability': 'Resilience'})
        temp_df['Network'] = file.split('_')[1]
        df = df.append(temp_df, ignore_index=True)
    return df



df = get_resilience('./results_1')
print(df.head())

#%%