import nltk
import random
from nltk.corpus import movie_reviews as mr

documents = [(list(mr.words(file_id)), category)
             for category in mr.categories()
             for file_id in mr.fileids(category)]

random.shuffle(documents)

all_words = []
for w in mr.words():
    all_words.append(w)

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(doc):
    words = set(doc)
    feature = {}
    for w in word_features:
        feature[w] = w in words
    return feature

print(find_features(mr.words('neg/cv000_29416.txt')))

feature_set = [(find_features(rev), category) for (rev, category) in documents]
print(feature_set)