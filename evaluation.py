import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
import pickle
import math
import random


test_size = 0.2
random_seed = 10
iterations = 10
p = 10  # rND
k = 10  # equity of attention
LOG_BASE = 2  # all
NORM_ITERATION = 10  # for rND
NORM_CUTPOINT = 10  # for rND
n_top_recommendations = 10
ranking_starts_at = 1  # IMPORTANT!

algo_names = ['Random', 'MostPop', 'UserKNN', 'ItemKNN', 'BPR', 'SVD']

df = pd.read_csv('data.csv')

with open("protected", "rb") as fp:  # Unpickling
    protected = list(pickle.load(fp))

protected = [int(item) for item in protected]

users_ids = list(df.user_id.unique())
users_ids = [str(user_id) for user_id in users_ids]
items_ids = list(df.book_code.unique())
items_ids = [int(item_id) for item_id in items_ids]
user_N = len(users_ids)
pro_N = len(set(items_ids) & set(protected))
N = len(items_ids)

unprotected = [int(item) for item in items_ids if item not in protected]


# discounted cumulative fairness

def discounted_cumulative_fairness(ranking, normalizer, name=None, i=None, save=True, protected=True):
    fairness_rND = defaultdict(list)

    for user_id in tqdm(users_ids):

        user_ranking = ranking[user_id]
        _ranking = []
        for item in user_ranking:
            _ranking.append(item[0])
        if protected:
            fairness_value = calculateNDFairness(_ranking, protected, p, normalizer)
            fairness_rND[user_id] = fairness_value
        else:
            fairness_value = calculateNDFairness(_ranking, unprotected, p, normalizer)
            fairness_rND[user_id] = fairness_value

    if save:
        with open('experiments/fairness/rND_{}_{}.json'.format(name, i), 'w') as json_file:
            json.dump(fairness_rND, json_file)

    print(np.mean(list(fairness_rND.values())))
    return fairness_rND


def calculateNDFairness(_ranking, _protected_group, _cut_point, _normalizer):
    discounted_gf = 0  # initialize the returned gf value
    for count in range(user_N):
        count = count + 1
        if count % _cut_point == 0:
            ranking_cutpoint = _ranking[0:count]
            pro_cutpoint = set(ranking_cutpoint).intersection(_protected_group)

            ranking_k = len(ranking_cutpoint)
            pro_k = len(pro_cutpoint)

            gf = calculateND(ranking_k, pro_k)
            discounted_gf += gf / math.log(count + 1, LOG_BASE)  # log base -> global variable

    if _normalizer == 0:
        print("Normalizer equals to zero")
    return discounted_gf / _normalizer


def calculateND(_ranking_k, _pro_k):
    return abs(_pro_k / _ranking_k - pro_N / user_N)


def generateUnfairRanking(input_ranking, protected_group, fpi):
    pro_ranking = [x for x in input_ranking if x not in protected_group]  # partial ranking of protected member
    unpro_ranking = [x for x in input_ranking if x in protected_group]  # partial ranking of unprotected member
    pro_ranking.reverse()  # prepare for pop function to get the first element
    unpro_ranking.reverse()
    unfair_ranking = []

    while len(unpro_ranking) > 0 and len(pro_ranking) > 0:
        seed = random.random()  # generate a random value in range [0,1]
        if seed < fpi:
            unfair_ranking.append(unpro_ranking.pop())  # insert protected group first
        else:
            unfair_ranking.append(pro_ranking.pop())  # insert unprotected group first

    if len(unpro_ranking) > 0:  # insert the remain unprotected member
        unpro_ranking.reverse()
        unfair_ranking = unfair_ranking + unpro_ranking
    if len(pro_ranking) > 0:  # insert the remain protected member
        pro_ranking.reverse()
        unfair_ranking = unfair_ranking + pro_ranking

    if len(unfair_ranking) < len(input_ranking):  # check error for insertation
        print("Error!")
    return unfair_ranking


def get_normalizer(protected=True):
    f_probs = [0, 0.98]
    avg_maximums = []

    for fpi in f_probs:
        iter_results = []  # initialize the lists of results of all iteration
        for iteri in range(NORM_ITERATION):
            input_ranking = [x for x in range(user_N)]
            if protected:
                protected_group = [x for x in range(pro_N)]
            else:
                protected_group = [x for x in range(N-pro_N)]
            # generate unfair ranking using algorithm
            unfair_ranking = generateUnfairRanking(input_ranking, protected_group, fpi)
            # calculate the non-normalized group fairness value i.e. input normalized value as 1
            gf = calculateNDFairness(unfair_ranking, protected_group, NORM_CUTPOINT, 1)
            iter_results.append(gf)
        avg_maximums.append(np.mean(iter_results))

    return max(avg_maximums)


# fairness of exposure

def fairness_of_exposure(ranking, name=None, i=None, top_n_check=False, save=True):
    exposure_prot = {}
    exposure_unprot = {}

    for user_id in tqdm(users_ids):
        user_ranking = ranking[user_id]
        user_ranking = [item[0] for item in user_ranking]
        prot = list(set(user_ranking) & set(protected))
        unprot = list(set(user_ranking) - set(prot))

        exposure_prot_user = 0
        for item in prot:
            item_ranking_index = user_ranking.index(item) + 1  # because we start ranking from 1 and not 0
            log_val = math.log(item_ranking_index + 1, LOG_BASE)
            exposure_prot_user += 1 / log_val

        exposure_unprot_user = 0
        for item in unprot:
            item_ranking_index = user_ranking.index(item) + 1
            log_val = math.log(item_ranking_index + 1, LOG_BASE)
            if log_val == 0:
                exposure_unprot_user += 1
            else:
                exposure_unprot_user += 1 / log_val

        exposure_prot[user_id] = exposure_prot_user / len(prot) if len(prot) > 0 else 0
        exposure_unprot[user_id] = exposure_unprot_user / len(unprot) if len(unprot) > 0 else 0

    print('prot', np.mean(list(exposure_prot.values())))
    print('unprot', np.mean(list(exposure_unprot.values())))

    if save:
        if top_n_check:
            with open('experiments/fairness/exposure_protected@10_{}_{}.json'.format(name, i), 'w') as json_file:
                json.dump(exposure_prot, json_file)
            with open('experiments/fairness/exposure_unprotected@10_{}_{}.json'.format(name, i), 'w') as json_file:
                json.dump(exposure_unprot, json_file)
        else:
            with open('experiments/fairness/exposure_protected_{}_{}.json'.format(name, i), 'w') as json_file:
                json.dump(exposure_prot, json_file)
            with open('experiments/fairness/exposure_unprotected_{}_{}.json'.format(name, i), 'w') as json_file:
                json.dump(exposure_unprot, json_file)

    return exposure_prot, exposure_unprot


# equity of attention


def calculate_geometric_position_bias(position, k):
    if position > k:
        return 0
    else:
        return 0.5 * (0.5 ** (position - 1))


def equity_of_attention(ranking, name, i, save=True):
    N = len(items_ids)
    m = len(users_ids)

    attention_prot = defaultdict()
    attention_unprot = defaultdict()

    unfairness_prot = 0
    unfairness_unprot = 0

    top_scores = defaultdict()
    min_scores = defaultdict()

    item_sum = 0
    for i in tqdm(range(N)):
        item = items_ids[i]
        attention_sum = 0
        relevance_sum = 0

        for j in range(m):
            ranking_j = ranking[users_ids[j]]
            items_j = [item[0] for item in ranking_j]
            index_item_j = items_j.index(item)
            scores_j = [item[1] for item in ranking_j]

            if users_ids in list(top_scores.keys()):
                top_score = top_scores[users_ids[j]]
                min_score = min_scores[users_ids[j]]
            else:
                top_score = max(scores_j)
                top_scores[users_ids[j]] = top_score
                min_score = min(scores_j)
                min_scores[users_ids[j]] = min_score

            attention_sum += calculate_geometric_position_bias(index_item_j + 1, k)
            relevance_sum += (scores_j[index_item_j] - min_score) / (top_score - min_score) if (
                                                                                                           top_score - min_score) != 0 else \
            scores_j[index_item_j]

        if item in protected:
            attention_prot[item] = attention_sum / m
            unfairness_prot += abs(attention_sum - relevance_sum)
        else:
            attention_unprot[item] = attention_sum / m
            unfairness_unprot += abs(attention_sum - relevance_sum)

        item_sum += abs(attention_sum - relevance_sum)

    unfairness = (1 / N) * (1 / m) * item_sum
    unfairness_prot = (1 / pro_N) * (1 / m) * unfairness_prot
    unfairness_unprot = (1 / pro_N) * (1 / m) * unfairness_unprot

    print(unfairness)
    print('prot', unfairness_prot)
    print('unprot', unfairness_unprot)

    if save:
        with open('experiments/fairness/attention_protected_{}_{}.json'.format(name, i), 'w') as json_file:
            json.dump(attention_prot, json_file)
        with open('experiments/fairness/attention_unprotected_{}_{}.json'.format(name, i), 'w') as json_file:
            json.dump(attention_unprot, json_file)

    return unfairness, unfairness_prot, unfairness_unprot, attention_prot, attention_unprot


def get_coverage(top_n):
    coverage_dict = {}
    for user_id in users_ids:
        user_top_n = top_n[user_id]
        coverage_user = 0
        for item in user_top_n:
            if item[0] in protected:
                coverage_user += 1
        coverage_dict[user_id] = coverage_user

    print(np.mean(list(coverage_dict.values())))
    return coverage_dict


def read_ranking(name, i):
    with open('experiments/rankings/{}_{}.json'.format(name, i)) as f:  # Unpickling
        ranking = json.load(f)
    return ranking


def read_top_n(name, i):
    with open('experiments/top_n/{}_{}.json'.format(name, i)) as f:  # Unpickling
        top_n = json.load(f)
    return top_n


def read_accuracy_values(name, i):
    with open('experiments/acc/{}_{}.json'.format(name, i)) as f:  # Unpickling
        acc = json.load(f)
    return acc


def evaluation():
    fairness_df = pd.DataFrame(columns=['model', 'iteration', 'metric', 'value', "time"])
    normalizer = get_normalizer(protected=False)

    for i in tqdm(range(iterations)):
        for name in tqdm(algo_names):
            print('  ', name, i)

            ranking = read_ranking(name, i)
            top_n = read_top_n(name, i)

            # rND
            start_time = time.time()
            dcf = discounted_cumulative_fairness(ranking, normalizer, name=name, i=i, protected=False)
            dcf_mean = np.mean(list(dcf.values()))
            end_time = time.time() - start_time
            fairness_df.loc[len(fairness_df)] = [name, i, 'rND_unprotected', dcf_mean, end_time]

            # exposure
            start_time = time.time()
            foe_p, foe_unp = fairness_of_exposure(ranking, name=name, i=i)
            foe_p_mean = np.mean(list(foe_p.values()))
            foe_unp_mean = np.mean(list(foe_unp.values()))
            end_time = time.time() - start_time
            fairness_df.loc[len(fairness_df)] = [name, i, 'exposure_protected', foe_p_mean, end_time]
            fairness_df.loc[len(fairness_df)] = [name, i, 'exposure_unprotected', foe_unp_mean, end_time]

            # exposure@10
            start_time = time.time()
            foe_p10, foe_unp10 = fairness_of_exposure(top_n, name=name, i=i, top_n_check=True)
            foe_p10_mean = np.mean(list(foe_p10.values()))
            foe_unp10_mean = np.mean(list(foe_unp10.values()))
            end_time = time.time() - start_time
            fairness_df.loc[len(fairness_df)] = [name, i, 'exposure@10_protected', foe_p10_mean, end_time]
            fairness_df.loc[len(fairness_df)] = [name, i, 'exposure@10_unprotected', foe_unp10_mean, end_time]

            # attention
            start_time = time.time()
            eoa, eoa_prot, eoa_unprot, a_prot, a_unprot = equity_of_attention(ranking, name, i)
            a_prot_mean = np.mean(list(a_prot.values()))
            a_unprot_mean = np.mean(list(a_unprot.values()))
            end_time = time.time() - start_time
            fairness_df.loc[len(fairness_df)] = [name, i, 'unfairness_of_attention', eoa, end_time]
            fairness_df.loc[len(fairness_df)] = [name, i, 'attention_protected', a_prot_mean, end_time]
            fairness_df.loc[len(fairness_df)] = [name, i, 'attention_unprotected', a_unprot_mean, end_time]
            fairness_df.loc[len(fairness_df)] = [name, i, 'unfairness_of_attention_protected', eoa_prot, end_time]
            fairness_df.loc[len(fairness_df)] = [name, i, 'unfairness_of_attention_unprotected', eoa_unprot, end_time]

            # coverage
            coverage = get_coverage(top_n)
            coverage_mean = np.mean(list(coverage.values()))
            fairness_df.loc[len(fairness_df)] = [name, i, 'coverage', coverage_mean, None]

            # other metrics
            acc = read_accuracy_values(name, i)

            # MAP
            fairness_df.loc[len(fairness_df)] = [name, i, 'MAP', acc['MAP'], None]

            # NDCG@10
            fairness_df.loc[len(fairness_df)] = [name, i, 'NDCG@10', acc['NDCG@10'], None]

            # Precision@10
            fairness_df.loc[len(fairness_df)] = [name, i, 'precision@10', acc['Precision@10'], None]

            #Recall@10
            fairness_df.loc[len(fairness_df)] = [name, i, 'recall@10', acc['Recall@10'], None]

        fairness_df.to_csv('experiments/fairness/evaluation.csv', index=False)

    fairness_df.to_csv('experiments/fairness/evaluation.csv', index=False)


evaluation()