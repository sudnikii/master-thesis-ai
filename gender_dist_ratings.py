import pandas as pd
from tqdm import tqdm
import pickle


iterations = 10
algo_names = ['Random', 'MostPop', 'UserKNN', 'ItemKNN', 'BPR', 'SVD']

with open("protected", "rb") as fp:  # Unpickling
    protected = list(pickle.load(fp))

protected = [int(item) for item in protected]


def read_ratings(name, i):
    df = pd.read_csv('experiments/ratings/{}_{}.csv'.format(name, i))
    return df


count_df = pd.DataFrame(columns=['name', 'i', 'female%'])

for i in tqdm(range(iterations)):
    for name in tqdm(algo_names[:1]):
        ratings = read_ratings(name, i)
        count_female = 0
        for item in tqdm(list(ratings.book_code)):
            if int(item) in protected:
                count_female += 1

        count_df.loc[len(count_df)] = [name, i, count_female/len(ratings)]

    count_df.to_csv('experiments/fairness/gender_dist_ratings_random.csv', index=False)



