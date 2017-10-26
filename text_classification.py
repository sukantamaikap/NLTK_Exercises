import nltk
import random
from nltk.corpus import movie_reviews as mr
import pickle

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


# print(find_features(mr.words('neg/cv000_29416.txt')))

feature_set = [(find_features(rev), category) for (rev, category) in documents]
print(feature_set)

training_set = feature_set[:1900]
testing_set = feature_set[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

# use the saved classifier only if it is saved
classifier_f = open('naive_bayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

print("accuracy percentage : ", nltk.classify.accuracy(classifier, testing_set) * 100)
classifier.show_most_informative_features(15)

# save_classifier = open('naive_bayes.pickle', 'wb')
# pickle.dump(classifier, save_classifier)
# save_classifier.close()
