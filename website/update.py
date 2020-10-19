import pickle
import sqlite3
import numpy as np
import os

# import HashingVectorizer from local dir
from amazonreview_vectorizer import vect

cur_dir = os.path.dirname(__file__)

def update_model(db_path, model, batch_size=10000):

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')

    results = c.fetchmany(batch_size)
    while results:
        inv_label = {'Negative': 0, 'Neutral':1, 'Positive':2}
        data = np.array(results)
        X = data[:, 0]
        for i in data:
            i[1] = int(inv_label[i[1]])

        y = data[:, 1].astype(int)
        classes = np.array([0, 1, 2])
        X_train = vect.transform(X)
        model.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()
    return model


clf = pickle.load(open(os.path.join(cur_dir,
                  'pkl_objects',
                  'amazon_classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')
clf = update_model(db_path=db, model=clf, batch_size=10000)
pickle.dump(clf, open(os.path.join(cur_dir,
             'pkl_objects', 'amazon_classifier.pkl'), 'wb')
             , protocol=4)
def update_model2(db_path, model, batch_size=10000):

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')

    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)
        classes = np.array([1,2,3,4,5])
        X_train = vect.transform(X)
        model.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()
    return model

clf2 = pickle.load(open(os.path.join(cur_dir,
          'pkl_objects',
          'amazon_classifier_2.pkl'), 'rb'))
db2 = os.path.join(cur_dir, 'reviews_2.sqlite')
clf2 = update_model2(db_path=db2, model=clf2, batch_size=10000)
pickle.dump(clf2, open(os.path.join(cur_dir,
             'pkl_objects', 'amazon_classifier_2.pkl'), 'wb')
             , protocol=4)
