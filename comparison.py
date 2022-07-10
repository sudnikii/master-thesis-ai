import pandas as pd
import json
from tqdm import tqdm

algo_names = ['Random', 'MostPop', 'UserKNN', 'ItemKNN', 'BPR', 'SVD']
metric_names = ['rND', 'exposure_protected', 'exposure_unprotected',
                'exposure@10_protected', 'exposure@10_unprotected',
                ]
iterations = [0, 4, 9]


def read_evaluation(metric, name, i):
    if metric == 'exposure@10_protected':
        metric = 'exposure_protected@10'
    elif metric == 'exposure@10_unprotected':
        metric = 'exposure_unprotected@10'

    with open('experiments/fairness/{}_{}_{}.json'.format(metric, name, i)) as f:  # Unpickling
        eval = json.load(f)
    return eval


comparison_df = pd.DataFrame(columns=['model', 'iteration', 'metric', 'value'])

for i in tqdm(iterations):
    for model in tqdm(algo_names):
        for metric in tqdm(metric_names):
            eval = read_evaluation(metric, model, i)
            for value in list(eval.values()):
                comparison_df.loc[len(comparison_df)] = [model, i+1, metric, value]

comparison_df.to_csv('experiments/fairness/comparison.csv')
