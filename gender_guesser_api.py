# gender-guesser 0.4.0 https://pypi.org/project/gender-guesser/
import gender_guesser.detector as gender
import pandas as pd
import time

books = pd.read_csv('books.csv')
d = gender.Detector()

start_time_part = time.time()

for index_label, row_series in books.iterrows():
    if index_label % 100 == 0:
        print(index_label, '/', len(books), '***', round(index_label / len(books), 5), '%')
        books.to_csv('books.csv', index=False)
        print("--- %s seconds ---" % (time.time() - start_time_part))
        start_time_part = time.time()

    if pd.isna(books.at[index_label, 'name']):
        pass

    name = str(books.at[index_label, 'name']).title()

    gender_label = d.get_gender(name)
    books.at[index_label, 'gender'] = gender_label

books.to_csv('books.csv', index=False)  # save the dataset
