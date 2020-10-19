from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# import update function from local dir
from update import update_model, update_model2 
# import HashingVectorizer from local dir
from amazonreview_vectorizer import vect

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
print(cur_dir)
db = os.path.join(cur_dir, 'reviews.sqlite')
db2 = os.path.join(cur_dir, 'reviews_2.sqlite')

clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'amazon_classifier.pkl'), 'rb'))
clf2 = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'amazon_classifier_2.pkl'), 'rb'))


def classify(document):
    label = {0: 'negative', 1: 'neutral', 2:'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def classify2(document):
    label = {1: 1, 2:2, 3:3, 4:4, 5:5}
    X = vect.transform([document])
    y = clf2.predict(X)[0]
    proba = np.max(clf2.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [int(y)])

def train2(document, y):
    X = vect.transform([document])
    clf2.partial_fit(X, [int(y)])

def sqlite_entry(path, document, y, p):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, actual_sentiment, predicted_sentiment, date)"\
    " VALUES (?, ?, ?, DATETIME('now'))", (document, y, p))
    conn.commit()
    conn.close()

def sqlite_entry2(path, document, y, s, p):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, actual_rating, actual_sentiment,predicted_rating, date)"\
    " VALUES (?, ?, ?,?,DATETIME('now'))", (document, y, s, p))
    conn.commit()
    conn.close()

def sqlite_view(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("SELECT * FROM review_db")
    items = c.fetchall()

    conn.commit()
    conn.close()

    return items

def sqlite_view2(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("SELECT * FROM review_db")
    items = c.fetchall()

    conn.commit()
    conn.close()

    return items


######## Flask
class ReviewForm(Form):
    amazonreview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/new')
def new_review():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/newrating')
def new_review_rating():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['amazonreview']
        y, proba = classify(review)
        y2, proba2 = classify2(review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2),
                                prediction2=y2,
                                probability2=round(proba2*100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    rating = request.form['rating']
    review = request.form['review']
    prediction = request.form['prediction']
    rating2 = request.form['rating2']
    prediction2 = request.form['prediction2']


    inv_label = {'negative': 0, 'neutral':1, 'positive':2}
    label = {0: 'Negative', 1:'Neutral', 2: 'Positive'}
    y = inv_label[prediction]
    if rating != y:
        y = rating
    train(review, y)
    sqlite_entry(db, review, label[int(rating)], prediction)

    label2 = {1: 'Negative', 2:'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}
    y2 = prediction2
    if rating2 != y2:
        y2 = rating2
    train2(review, y2)
    sqlite_entry2(db2, review, int(rating2), label2[int(rating2)], label2[int(prediction2)])
    return render_template('home.html')

@app.route('/view')
def view_reviews():
    items = sqlite_view(db)
    return render_template('view_reviews.html', items=items)

@app.route('/viewrating')
def view_reviews2():
    items = sqlite_view2(db2)
    return render_template('view_reviews2.html', items=items)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
