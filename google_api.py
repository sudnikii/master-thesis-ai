import pandas as pd
import time
import urllib.request
import json


def google_api(isbn):
    base_api_link = "https://www.googleapis.com/books/v1/volumes?q=isbn:"

    with urllib.request.urlopen(base_api_link + isbn) as f:
        text = f.read()

    decoded_text = text.decode("utf-8")
    obj = json.loads(decoded_text)  # deserializes decoded_text to a Python object

    if "items" in obj.keys():
        volume_info = obj["items"][0]["volumeInfo"]

        title = volume_info["title"].lower() if "title" in volume_info.keys() else None
        author = volume_info["authors"][0].lower() if "authors" in volume_info.keys() else None
        language = volume_info["language"] if "language" in volume_info.keys() else None

        return title, author, language

    return None, None, None


books = pd.read_csv('books.csv')

start_time = time.time()

for index_label, row_series in books[books.title_google.isna()].iterrows():
    try:
        if index_label % 100 == 0:
            print(index_label, '/', len(books[books.title_google.isna()]), '***', round(index_label / len(books), 5), '%')
            books.to_csv('books.csv', index=False)  # save the dataset
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

        title, author, language = google_api(row_series.isbn)

        # print(row_series.isbn, title, author, language)

        books.at[index_label, 'title_google'] = title
        books.at[index_label, 'author_google'] = author
        books.at[index_label, 'language'] = language

    except Exception:
        pass


books.to_csv('books.csv', index=False)  # save the dataset
