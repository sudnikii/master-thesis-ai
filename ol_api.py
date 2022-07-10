import urllib.request
import time
import pandas as pd
import json


def ol_api(isbn):
    base_api_link = 'https://openlibrary.org/isbn/'
    with urllib.request.urlopen(base_api_link + isbn + '.json') as f:
        text = f.read()

    decoded_text = text.decode("utf-8")
    obj = json.loads(decoded_text)  # deserializes decoded_text to a Python object
    author_key = obj["authors"][0]['key'].split("authors/", 1)[1] if "authors" in obj.keys() else None
    book_key = obj["works"][0]['key'].split("works/", 1)[1] if "works" in obj.keys() else None

    return author_key, book_key


books = pd.read_csv('books.csv')

start_time = time.time()

for index_label, row_series in books.iterrows():
    try:
        author_key, book_key = ol_api(row_series.isbn)
        books.at[index_label, 'author_key_ol'] = author_key
        books.at[index_label, 'book_key_ol'] = book_key

        if index_label % 100 == 0:
            print(index_label, '/', len(books), '***', round(index_label / len(books), 5), '%')
            books.to_csv('books.csv', index=False)  # save the dataset
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

    except Exception:
        pass

books.to_csv('books.csv', index=False)  # save the dataset
