#!/usr/bin/env python

import warnings
import random as rd
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json

from cornac.eval_methods import RatioSplit
from cornac.data import Reader as CornacReader
from cornac.models import MostPop, BPR, UserKNN, ItemKNN, SVD
from cornac.metrics import Precision, Recall, NDCG, MAP

pd.set_option("display.precision", 6)
warnings.simplefilter(action='ignore', category=FutureWarning)


# ### set parameters

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def create_all_paths():
    paths = {}
    for model in algo_names:
        paths[model] = {}
        for i in range(iterations):
            path = 'experiments/ratings/{}_{}.csv'.format(model, i)
            paths[model][i] = path
    return paths


test_size = 0.2
random_seed = 10
iterations = 10
n_top_recommendations = 10
algo_names = ['Random', 'MostPop', 'UserKNN', 'ItemKNN', 'BPR', 'SVD']
paths = create_all_paths()
df = pd.read_csv('data.csv')
users_ids = list(df.user_id.unique())
users_ids = [str(user_id) for user_id in users_ids]
items_ids = list(df.book_code.unique())
items_ids = [int(item_id) for item_id in items_ids]

# initialize models, here we are comparing: simple, traditional, and neural networks based models
models = [
    # 1: Random
    None,
    # 2: MostPop
    MostPop(),
    # 3: UserKNN
    UserKNN(seed=123),
    # 4: UserKNN
    ItemKNN(seed=123),
    # 5: BPR
    BPR(seed=123),
    # 6: SVD
    SVD(seed=123)
]

# define metrics to evaluate the models
metrics = [MAP(), Precision(k=n_top_recommendations),
           Recall(k=n_top_recommendations), NDCG(k=n_top_recommendations),
           ]


# #### random

def get_ranking_random(name, i, save=True):
    ranking = defaultdict(list)
    for user_id in users_ids:
        possible_items_set = set(items_ids)
        for j in range(0, len(possible_items_set)):
            el = rd.sample(possible_items_set,1)[0]
            ranking[int(user_id)].append((el, j))
            possible_items_set.remove(el)
    if save:
        with open('experiments/rankings/{}_{}.json'.format(name, i), 'w') as json_file:
            json.dump(ranking, json_file)

    return ranking


def get_top_n_random(name, i, save=True):
    print("Random model is selected:")
    top_n = defaultdict(list)
    for user_id in users_ids:
        for j in range(0, n_top_recommendations):
            top_n[int(user_id)].append((int(rd.choice(items_ids)), j))

    if save:
        with open('experiments/top_n/{}_{}.json'.format(name, i), 'w') as json_file:
            json.dump(top_n, json_file)

    return top_n


# #### all

def get_ranking(rs, model, name, i, save=True):
    ranking = defaultdict(list)

    for uid in tqdm(rs.train_set.uid_map.values()):
        user_id = list(rs.train_set.user_ids)[uid]

        # full ranking
        try:
            item_rank = model.rank(user_idx=uid)[0]  # model.rank: item rank, item_score
        except:
            item_rank = model.rank(user_idx=int(uid))[0]

        for iid in item_rank:
            item_id = list(rs.train_set.item_ids)[iid]  # change from Cornac id to book id
            ranking[int(user_id)].append((int(item_id), model.score(uid, iid)))

    if save:
        with open('experiments/rankings/{}_{}.json'.format(name, i), 'w') as json_file:
            json.dump(ranking, json_file, cls=NpEncoder)

    return ranking


def get_top_new(ranking, rs, name, i, save=True):
    top_n = defaultdict(list)

    for uid in tqdm(rs.train_set.uid_map.values()):
        user_id = list(rs.train_set.user_ids)[uid]

        # history only based on the train set
        history = [list(rs.train_set.item_ids)[int(iid)] for iid in list(rs.train_set.user_data[uid][0])]
        history = [int(item) for item in history]

        # user ranking
        user_ranking = ranking[int(user_id)]

        position = 0
        top_n_list = []

        # collect top N items (make sure it's not duplicated)
        while len(top_n_list) < n_top_recommendations:
            if int(user_ranking[position][0]) in history:
                position += 1
            else:
                top_n_list.append(user_ranking[position])
                position += 1

        top_n[user_id] = top_n_list

    if save:
        with open('experiments/top_n/{}_{}.json'.format(name, i), 'w') as json_file:
            json.dump(top_n, json_file, cls=NpEncoder)

    return top_n


def update_history(df, top_n, name, i, save=True):
    user_ids = []
    items = []
    scores = []

    for user in top_n.keys():
        for item, score in top_n[user]:
            user_ids.append(user)
            items.append(item)
            scores.append(1)

    ratings_update = pd.DataFrame({'user_id': user_ids, 'book_code': items, 'rating': scores})
    ratings_update = ratings_update.append(df)  # add to history of all ratings
    ratings_update = ratings_update.drop_duplicates()  # drop duplicates from test set

    # save file
    if save:
        ratings_update.to_csv('experiments/ratings/{}_{}.csv'.format(name, i + 1), index=False)
    pass


def create_starting_files(df):
    for model in algo_names:
        df.to_csv('experiments/ratings/{}_0.csv'.format(model), index=False)


def evaluate_accuracy(rs, model, name, i, save=True):
    test_result, val_result = rs.evaluate(
        model=model, metrics=metrics, user_based=True
    )

    print(test_result)

    if save:
        with open('experiments/acc/{}_{}.json'.format(name, i), 'w') as json_file:
            json.dump(test_result.metric_avg_results, json_file, cls=NpEncoder)


def run_experiment(i):
    for name in tqdm(algo_names[:1], desc='model runs'):
        path = paths[name][i]
        ratings = pd.read_csv(path)

        reader = CornacReader()
        data = reader.read(path, sep=",", skip_lines=1)
        rs = RatioSplit(data=data, test_size=test_size, seed=123)

        if name == 'Random':
            # random
            print('  ', name + " model is selected:")
            print("- Training the model finished.")
            get_ranking_random(name, i)
            print("- Ranking created.")
            top_n = get_top_n_random(name, i)
            print("- Top recommendations calculated.")
            update_history(ratings, top_n, name, i)
            print("- Rating file updated.")

        else:
            # cornac
            print('  ', name + " model is selected:")
            model = models[algo_names.index(name)]
            model.fit(rs.train_set)
            print("- Training the model finished.")
            ranking = get_ranking(rs, model, name, i)
            print("- Ranking created.")
            top_n = get_top_new(ranking, rs, name, i)
            print("- Top recommendations calculated.")
            evaluate_accuracy(rs, model, name, i)
            print("- Accuracy evaluated.")
            update_history(ratings, top_n, name, i)
            print("- Rating file updated.")
            print('Saving the model...')
            model.save('experiments/models/{}_{}'.format(name, i))
            print('Model saved.')

            del model

        del path
        del ratings
        del data
        del rs
        del top_n


def main():
    df = pd.read_csv('data.csv')
    create_starting_files(df)
    print('Starting files created.')

    for i in tqdm(range(iterations), desc='iterations'):
        run_experiment(i)


main()

