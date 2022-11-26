import pandas as pd
import numpy as np
from os import path
import seaborn as sns
import matplotlib.pyplot as plt
import os

aspects = ['Service', 'Spatial']
dimension = ['Magnitude', 'Peak', 'Time to Peak', 'Time to Recovery']
path_result = 'results/reliability'
reliability_data = []


for filename in os.listdir(path_result):
    df = pd.read_csv(os.path.join(path_result, filename), index_col=0)

    print(df)
    break
