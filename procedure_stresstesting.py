import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr
import pickle
import seaborn as sns
import os.path
from os import path
sns.set_style("whitegrid")

class substance_intrusion():
    def __init__(self, inp, mass_value=10, duration=1, cs_e=50, cs_p=100):
        '''
        :arg
        inp: Epanet .inp file
        duration: [min, max]
        start_time: [min, max]
        mass_value: MASS [mg/s] at booster point
        cs_e: Threshold for low quality water supply [mg/L]
        cs_p: Threshold for polluted water supply [mg/L]
        '''

        self.cs_e = cs_e
        self.cs_p = cs_p
        self.duration = duration
        self.mass_value = mass_value
        self.network_file = 'wn.pickle'
        self.inp = inp
        self.quality_timestep = 60
        self.hydraulic_timestep = 3600
        self.report_timestep = 3600
        self.simulation_duration = 24 * 3600 * 4


        wn = wntr.network.WaterNetworkModel(self.inp)
        wn.options.hydraulic.demand_model = 'DD'
        wn.options.quality.parameter = 'CHEMICAL'
        wn.options.time.duration = self.simulation_duration
        wn.options.time.hydraulic_timestep = self.hydraulic_timestep
        wn.options.time.report_timestep = self.report_timestep
        wn.options.time.quality_timestep = self.quality_timestep

        self.junction_name_list = wn.junction_name_list
        self.demand = wntr.metrics.expected_demand(wn)[self.junction_name_list]
        self.total_nodes = len(self.junction_name_list)
        self.total_time = self.simulation_duration

        with open(self.network_file, 'wb') as f:
            pickle.dump(wn, f)

    def random_failure_sequences(self, x=1, n=10):
        result = pd.DataFrame(columns=['Simulation', 'Dimension', 'Aspect', 'Threshold', 'Reliability', 'Stress'])
        stress = stress = np.logspace(0.1, 2, n, endpoint=True)/100
        elements = [int(s * self.total_nodes) for s in stress]

        for i in range(x):
            print(f'{variant}: {i+1} of 300')
            for s in elements:
                pool = list(np.random.choice(self.junction_name_list, s, replace=False))
                reliability = self.sim_failure(sources=pool)[-1]
                reliability['Simulation'] = i + 1
                reliability['Stress'] = s * 100 / self.total_nodes
                result = result.append(reliability, ignore_index=True)

        reliability = reliability.copy(deep=True)
        reliability['Stress'] = 0
        reliability['Simulation'] = 0
        reliability['Reliability'] = 100
        result = result.append(reliability, ignore_index=True)
        return result

    def sim_failure(self, sources, start=0, duration=1, mass=None):
        with open(self.network_file, 'rb') as f:
            wn = pickle.load(f)

        if mass is not None: mass = self.mass_value
        if type(sources) is not list: sources = [sources]

        source_pattern = wntr.network.elements.Pattern.binary_pattern(
            'SP', start_time=start * self.report_timestep,
            end_time=(start + duration) * self.report_timestep,
            duration=wn.options.time.duration,
            step_size=wn.options.time.pattern_timestep
        )
        wn.add_pattern('SP', source_pattern)

        for source in sources:
            wn.add_source( f'Source_{source}', source, 'MASS', self.mass_value, 'SP')

        try:
            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()
        except:
            print('Epanet Error')
            return False

        supply = results.node['demand'].loc[:, wn.junction_name_list]
        concentration = results.node['quality'].loc[:, wn.junction_name_list]
        return self._reliability(supply, concentration, start)

    def plot_resilience(self, reliability, aspect, dimension='Magnitude', threshold=1):
        df = reliability[reliability['Threshold'] == threshold]
        df = df[df['Aspect'] == aspect]
        df = df[df['Dimension'] == dimension]
        sns.scatterplot(x=df['Stress'], y=df['Reliability'])
        sns.lineplot(x=df['Stress'], y=df['Reliability'], alpha=0.5)
        plt.yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
        plt.xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
        plt.title(f'{dimension} of {aspect} Dimension, Threshold = {threshold} ')
        plt.show()

    def _reliability(self, supply, concentration, start):
        N = {0: concentration > self.cs_e, 1: concentration > self.cs_p}
        T, PS = {}, {}
        for k in N.keys():
            T[k] = (N[k].sum(axis=1) > 0)
            PS[k] = supply.where(N[k], other=0).sum(axis=1)

        data = pd.DataFrame(columns=['Dimension', 'Aspect', 'Threshold', 'Reliability'])
        for i in N.keys():
            #Magnitude
            T_sum = T[i].sum()
            PS_sum = PS[i].sum()
            N_sum = (N[i].sum(axis=0) > 0).astype(int).sum()
            N[i] = N[i].astype(int).sum(axis=1)

            data.loc[r'$PS_{{L{0}, %}}$'.format(i)] = ['Magnitude', 'Service', i, (1 - PS_sum/ supply.sum().sum()) * 100]
            data.loc[r'$N_{{L{0}, n, %}}$'.format(i)] = ['Magnitude', 'Spatial', i, (1 - N_sum / self.total_nodes) * 100]
            data.loc[r'$T_{{L{0}, t, %}}$'.format(i)] =  ['Magnitude', 'Continuity', i, (1 - T_sum * self.report_timestep / self.total_time) * 100]

            # Peak
            PS_max = PS[i].max()
            S_max = supply.sum(axis=1).max()
            N_max =  N[i].max()
            data.loc[r'$PS_{{L{0}, peak}}$'.format(i)] =  ['Peak', 'Service', i,  (1 - PS_max / S_max) * 100]
            data.loc[r'$N_{{L{0}, peak}}$'.format(i)] = ['Peak', 'Spatial', i, (1 - N_max / self.total_nodes) * 100]

            t_start = start * self.report_timestep
            t_total = self.total_time - t_start

            # Service rapdity
            if PS_max > 0:
                data.loc[r'$PS_{{L{0}, TEP}}$'.format(i)] =  ['Time to Peak', 'Service', i, ((PS[i].idxmax() - t_start) / t_total) * 100]
                if PS[i].iloc[-1] == 0:
                    tec = (1 - (PS[i][PS[i] != 0].index[-1] + self.report_timestep - t_start)/t_total) * 100
                    data.loc[r'$PS_{{L{0}, TEC}}$'.format(i)] = ['Time to Recovery', 'Service', i, tec]
                else:
                    data.loc[r'$PS_{{L{0}, TEC}}$'.format(i)] = ['Time to Recovery', 'Service', i, 0]
            else:
                data.loc[r'$PS_{{L{0}, TEP}}$'.format(i)] = ['Time to Peak', 'Service', i, 100]
                data.loc[r'$PS_{{L{0}, TEC}}$'.format(i)] = ['Time to Recovery', 'Service', i, 100]

            # Spatial rapidity
            if N_max > 0:
                data.loc[r'$N_{{L{0}, TEP}}$'.format(i)] = ['Time to Peak', 'Spatial', i,  ((N[i].idxmax() -  t_start) / t_total) * 100]

                if N[i].iloc[-1] == 0:
                    tec = (1 - (N[i][N[i] != 0].index[-1] + self.report_timestep - t_start)/t_total) * 100
                    data.loc[r'$N_{{L{0}, TEC}}$'.format(i)] = ['Time to Recovery', 'Spatial', i, tec]
                else:
                    data.loc[r'$N_{{L{0}, TEC}}$'.format(i)] = ['Time to Recovery', 'Spatial', i, 0]
            else:
                data.loc[r'$N_{{L{0}, TEP}}$'.format(i)] = ['Time to Peak', 'Spatial', i, 100]
                data.loc[r'$N_{{L{0}, TEC}}$'.format(i)] = ['Time to Recovery', 'Spatial', i, 100]

        return PS, N, T, data

n = 0
for i in range(300):
    variant = f'variants/variant_{n}.inp'
    if path.exists(variant):
        wn = substance_intrusion(variant)
        result = wn.random_failure_sequences(x=300)
        result.to_csv(f'results_2/Ctown_{n}_reliability.csv')
    else:
        print(variant + ' dont exists')
    n += 1


