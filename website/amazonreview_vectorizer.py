from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
import re
import os

porter=PorterStemmer()

cur_dir = os.path.dirname(__file__)


def tokenizer(text):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text.split()

def tokenizerporter(text):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return [porter.stem(word) for word in text.split()]

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         ngram_range=(1,1),
                         stop_words=None,
                         tokenizer=tokenizer)
