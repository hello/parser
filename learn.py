import sys
import pandas

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt


def tokenize(text):
    text = unicode(text, 'utf-8')
    return TextBlob(text).words

def lemma_it(text):
    text = unicode(text, 'utf-8')
    words = TextBlob(text).words
    lemmas = [word.lemma for word in words]
    return lemmas

def vectorize(text, bow, tfidf):
    text_bow = bow.transform(text)
    text_tfidf = tfidf.transform(text_bow)
    return text_tfidf

def predict(classifier, features, labels):
    predictions = classifier.predict(features)
    print predictions

    print 'accuracy', accuracy_score(labels, predictions)
    print 'confusion matrix\n', confusion_matrix(labels, predictions)
    print '(row=expected, col=predicted)'

    print classification_report(labels, predictions)
    
    if hasattr(classifier, 'predict_proba'):
        prob = classifier.predict_proba(features)
    else:
        prob_pos = classifier.decision_function(features)
        prob = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min()) # normalize

    return predictions, prob

def plot_cm(labels, predictions):
    # plot confusion matrix
    plt.matshow(confusion_matrix(labels, predictions),
        cmap=plt.cm.binary, interpolation='nearest')
    plt.title('confusion matrix')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')


def main(training_file, testing_file):
    data = pandas.read_csv(training_file, sep=",", header=1, names=['label', 'text'])
    print data[:10]

    data.groupby('label').describe()

    bow_transformer = CountVectorizer(#analyzer=lemma_it,
        ngram_range=(1,3),
        token_pattern=r'\b\w+\b',
        min_df=1).fit(data['text'])
    print bow_transformer.get_feature_names()
    data_bow = bow_transformer.transform(data['text'])

    tfidf_transformer = TfidfTransformer().fit(data_bow)
    data_tfidf = tfidf_transformer.transform(data_bow)
    print data_tfidf.shape

    # train a model and predict
    nb_predictor = MultinomialNB().fit(data_tfidf, data['label'])
    data['nb'], prob_nb = predict(nb_predictor, data_tfidf, data['label'])

    svc_predictor = LinearSVC(C=1.0).fit(data_tfidf, data['label'])
    data['svc'], prob_svc = predict(svc_predictor, data_tfidf, data['label'])

    print_errors(data, {'nb': prob_nb, 'svc': prob_svc})

    print "Done Training"

    if testing_file is not None:
        print "Start Testing"
        test_data = pandas.read_csv(testing_file, sep=",", header=1, names=['label', 'text'])
        test_features = vectorize(test_data['text'], bow_transformer, tfidf_transformer)
        test_data['nb'], pnb = predict(nb_predictor, test_features, test_data['label'])
        test_data['svc'], psvc = predict(svc_predictor, test_features, test_data['label'])
        print_errors(test_data, {'nb': pnb, 'svc': psvc})


def print_errors(data, probs):
    for i in data.index:
        label = data['label'][i]
        nb = data['nb'][i]
        svc = data['svc'][i]
        if label != nb:
            print "i=%00d, true=%d, nb=%d %0.4f %0.4f, svc=%d %0.4f svc_nb=%0.4f text=%s" % (
            i, label, 
            nb, probs['nb'][i][label], probs['nb'][i][nb],
            svc, probs['svc'][i][label], probs['svc'][i][nb],
            data['text'][i])

if __name__ == "__main__":
    training = sys.argv[1]
    testing = None
    if len(sys.argv) == 3:
        testing = sys.argv[2]
    main(training, testing)
