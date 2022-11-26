import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
import os
import re

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

def get_attributes(file):
    df = pd.read_csv(file)
    arr = list()
    columns = df.columns.to_list()[1:]
    for n, row in df.iterrows():
        net_name = row['Variant'].split('_')[-1]
        for col in columns:
            arr.append([net_name, col, row[col]])
    return pd.DataFrame(arr, columns=['Network', 'Parameter', 'Value'])

resilience = get_resilience('./reliability')
attributes = get_attributes('attributes.csv')
attributes = attributes[attributes['Network'].isin(resilience['Network'].unique())]



#%%

def scatterplot(dim, aspect, att_list):
    sns.set_style("whitegrid")
    condition = (resilience['Dimension'] == dim) & (resilience['Aspect'] == aspect)
    res_series = resilience[condition][['Network', 'Resilience', 'Threshold']]

    objs = []
    for att in att_list:

        att_series = attributes[attributes['Parameter'] == att][['Network', 'Value']]
        att_series = att_series.set_index('Network')
        df = res_series.set_index('Network')
        df['Value'] = att_series['Value']
        df['Attribute'] = att
        objs.append(df)


    df = pd.concat(objs)


    sns.set_context("paper", font_scale=1.5)
    #sns.set(rc={"figure.figsize":(15, 10)})
    sns.set_style("white")
    sns.set_palette("bright")

    g = sns.lmplot(data=df, x='Value', y='Resilience', hue='Threshold', col='Attribute', sharex=False, legend_out=False)
    plt.savefig('scatter_corr.png', bbox_inches='tight')
    plt.close()

def scatter_facet(threshold, aspect):

    condition = (resilience['Threshold'] == threshold) & (resilience['Aspect'] == aspect)
    df_res = resilience[condition][['Network', 'Dimension', 'Resilience']]
    df_att = attributes[['Network', 'Parameter', 'Value']]

    arr = list()
    for n, att_row in df_att.iterrows():
        net = att_row['Network']
        for i, res_row in df_res[df_res['Network'] == net].iterrows():
            arr.append([net, att_row['Parameter'], att_row['Value'], res_row['Dimension'], res_row['Resilience']])


    df = pd.DataFrame(arr, columns=['Network', 'Att', 'Attribute value', 'Res', 'Resilience score'])
    return df

#df = scatter_facet(1, 'Service')
scatterplot('Magnitude', 'Service', ['Link density', 'Bridge density'])



#%%

scatterplot('Magnitude', 'Service', ['Link density', 'Bridge density', 'Central Point of Dominance'])



#%%

def corr_data(corr, threshold):
    corr = corr.sort_values(by='r-value')
    corr = corr[(corr['Resilience threshold'] == threshold) & (corr['Resilience aspect'] != 'Continuity')]
    corr_dict = {}
    rename = {('Service', 'Magnitude'): 'S',
              ('Service', 'Peak'): r'$S_{peak}$',
              ('Service', 'Time to Recovery'): r'$S_{TER}$',
              ('Service', 'Time to Peak'): r'$S_{TEP}$',
              ('Spatial', 'Magnitude'): 'N',
              ('Spatial', 'Peak'): r'$N_{peak}$',
              ('Spatial', 'Time to Recovery'): r'$N_{TER}$',
              ('Spatial', 'Time to Peak'): r'$N_{TEP}$'
              }
    for dim in corr['Resilience dimension'].unique():
        for aspect in corr['Resilience aspect'].unique():
            key = rename[(aspect, dim)]
            corr_dict[key] = {}
            for att in corr['Attribute'].unique():
                condition = (corr['Resilience dimension'] == dim) \
                            & (corr['Resilience aspect'] == aspect) \
                            & (corr['Attribute'] == att)

                corr_dict[key][att] = corr[condition]['r-value'].to_list()[0]
    return pd.DataFrame(corr_dict)
def pearson():
    import matplotlib
    font = {'size': 17}
    matplotlib.rc('font', **font)

    columns = ['Resilience dimension', 'Resilience aspect', 'Resilience threshold', 'Attribute', 'r-value', 'p-value']

    arr = []
    for dim in resilience['Dimension'].unique():
        for aspect in resilience['Aspect'].unique():
            for threshold in resilience['Threshold'].unique():
                for att in attributes['Parameter'].unique():
                    res_condition =  (resilience['Threshold'] == threshold) & \
                                     (resilience['Aspect'] == aspect) & \
                                     (resilience['Dimension'] == dim)
                    res_series = resilience[res_condition][['Resilience', 'Network']].sort_values(by='Network')
                    att_series = attributes[attributes['Parameter'] == att][['Value', 'Network']].sort_values(by='Network')

                    x = res_series['Resilience'].to_list()
                    y = att_series['Value'].to_list()
                    if len(x) == 0 or len(y) == 0:
                        continue
                    r, p = stats.pearsonr(x, y)
                    arr.append([dim, aspect, threshold, att, r, p])
    corr = pd.DataFrame(arr, columns=columns)


    corr_0 = corr_data(corr, 0)
    corr_1 = corr_data(corr, 1)
    return corr_0, corr_1


corr_0, corr_1 = pearson()

rename = {'Link density': 'd',
           'Average shortest path length': 'a',
            'Graph diameter': r'$C_c$',
           'Clustering Coefficient': r'$C_d$',
           'Graph radius': r'$G_d$',
           'Central Point of Dominance': r'$G_r$',
           'Bridge density': r'$B_d$'}

corr_0 = corr_0.rename(index=rename)
sns.set_context("paper", font_scale=2)
plt.rcParams['figure.figsize'] = [15, 8]
sns.set_style("white")
sns.set_palette("bright")


f, (ax1, ax2, axcb) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]})
ax1.get_shared_y_axes().join(ax2, )
g1 = sns.heatmap(corr_0, vmin=-0.6, vmax=0.6, cmap='bwr', cbar=False, ax=ax1)
g1.set_title('Threshold = 0')
g1.set_ylabel('')
g1.set_xlabel('')
g2 = sns.heatmap(corr_1, vmin=-0.6, vmax=0.6, cmap='bwr', cbar_ax=axcb, ax=ax2)
g2.set_title('Threshold = 1')
g2.set_ylabel('')
g2.set_xlabel('')
g2.set_yticks([])
# may be needed to rotate the ticklabels correctly:
for ax in [g1, g2]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('pearson_nr.png', bbox_inches='tight')
plt.close()











