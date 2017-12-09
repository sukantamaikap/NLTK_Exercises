import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    """ Classifier to choose from multiple models """

    def labels(self):
        pass

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, featureset):
        votes = []
        for c in self._classifiers:
            v = c.classify(featureset)
            votes.append(v)
        return mode(votes)

    def confidence(self, featureset):
        votes = []
        for classifier in self._classifiers:
            v = classifier.classify(featureset)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_positive = open('dataSet/positive.txt', 'r').read()
short_negative = open('dataSet/negative.txt', 'r').read()

documents = []
all_words = []
# fish only the adjectives
allowed_word_types = ['J', 'R', 'V']

try:
    documents = pickle.load(open("~/pickles/documents.pickle","rb"))
    print('found pickle for documents')
except (OSError, IOError) as e:
    print("pickle not found, will regenerate documents")
    for r in short_positive.split('\n'):
        documents.append((r, 'pos'))
        words = nltk.word_tokenize(r)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    for r in short_negative.split('\n'):
        documents.append((r, 'neg'))
        words = nltk.word_tokenize(r)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    save_documents = open("~/pickles/documents.pickle","wb")
    pickle.dump(documents, save_documents)
    save_documents.close()

short_positive_words = nltk.word_tokenize(short_positive)
short_negative_words = nltk.word_tokenize(short_negative)

# for w in short_positive_words:
#     all_words.append(w.lower())
#
# for w in short_negative_words:
#     all_words.append(w.lower())
#

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]
save_word_features = open('~/pickles/word_features5K.pickle', 'wb')
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(doc):
    words = nltk.word_tokenize(doc)
    feature = {}
    for w in word_features:
        feature[w] = w in words
    return feature


feature_set = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(feature_set)
print(feature_set)

training_set = feature_set[:10000]
testing_set = feature_set[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

# use the saved classifier only if it is saved
classifier_f = open('naive_bayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

print("naive bayes accuracy percentage : ", nltk.classify.accuracy(classifier, testing_set) * 100)
classifier.show_most_informative_features(15)

save_classifier = open('naive_bayes.pickle', 'wb')
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB classifier accuracy percentage : ", nltk.classify.accuracy(MNB_classifier, testing_set) * 100)

# GMB_classifier = SklearnClassifier(GaussianNB())
# GMB_classifier.train(training_set)
# print("GMB classifier accuracy percentage : ", nltk.classify.accuracy(GMB_classifier, testing_set) * 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB classifier accuracy percentage : ",
      nltk.classify.accuracy(BernoulliNB_classifier, testing_set) * 100)

LogisticRegression_classifier = SklearnClassifier(BernoulliNB())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression classifier accuracy percentage : ",
      nltk.classify.accuracy(LogisticRegression_classifier, testing_set) * 100)

SGDC_classifier = SklearnClassifier(BernoulliNB())
SGDC_classifier.train(training_set)
print("SGDClassifier classifier accuracy percentage : ",
      nltk.classify.accuracy(SGDC_classifier, testing_set) * 100)

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

SVC_classifier = SklearnClassifier(BernoulliNB())
SVC_classifier.train(training_set)
print("SVC classifier accuracy percentage : ", nltk.classify.accuracy(SVC_classifier, testing_set) * 100)

LinearSVC_classifier = SklearnClassifier(BernoulliNB())
LinearSVC_classifier.train(training_set)
print("LinearSVC classifier accuracy percentage : ", nltk.classify.accuracy(LinearSVC_classifier, testing_set) * 100)

NuSVC_classifier = SklearnClassifier(BernoulliNB())
NuSVC_classifier.train(training_set)
print("NuSVC classifier accuracy percentage : ", nltk.classify.accuracy(NuSVC_classifier, testing_set) * 100)

voted_classifier = VoteClassifier(MNB_classifier, classifier, BernoulliNB_classifier, SGDC_classifier,
                                  SVC_classifier, LinearSVC_classifier, NuSVC_classifier)
print("voted classifier accuracy percentage : ", nltk.classify.accuracy(voted_classifier, testing_set) * 100)
print("Classification : ", voted_classifier.classify(testing_set[0][0]), " Confidence : ",
      voted_classifier.confidence(testing_set[0][0]))
print("Classification : ", voted_classifier.classify(testing_set[1][0]), " Confidence : ",
      voted_classifier.confidence(testing_set[1][0]))
print("Classification : ", voted_classifier.classify(testing_set[2][0]), " Confidence : ",
      voted_classifier.confidence(testing_set[2][0]))
print("Classification : ", voted_classifier.classify(testing_set[3][0]), " Confidence : ",
      voted_classifier.confidence(testing_set[3][0]))
print("Classification : ", voted_classifier.classify(testing_set[4][0]), " Confidence : ",
      voted_classifier.confidence(testing_set[4][0]))
print("Classification : ", voted_classifier.classify(testing_set[5][0]), " Confidence : ",
      voted_classifier.confidence(testing_set[5][0]))
print("Classification : ", voted_classifier.classify(testing_set[6][0]), " Confidence : ",
      voted_classifier.confidence(testing_set[6][0]))
print("Classification : ", voted_classifier.classify(testing_set[7][0]), " Confidence : ",
      voted_classifier.confidence(testing_set[7][0]))

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats)
