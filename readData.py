import pandas as pd
from bs4 import BeautifulSoup as bs
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Preprocess a given review.
def from_review_to_words(review):
    # Remove tags from one movie review
    noTag = bs(review)

    # Remove punctuation, change to lower case and split into words
    letter_only = re.sub("[^a-zA-Z]", " ", noTag.get_text()).lower().split()

    # Convert stopword dictionary to set.
    stops = set(stopwords.words("english"))

    # Remove stop words
    removeStop = [w for w in letter_only if not w in stops]

    # Convert back to sentence and return it
    return (" ".join(removeStop))

# Read in the data
train = pd.read_csv("data/trainingSample.tsv", header=0, \
                        delimiter = "\t", quoting=3)

# List for the clean reviews
clean_reviews = []

#  Clean the reviews and append to clean_reviews
for i, review in enumerate(train["review"]):
    clean_reviews.append(from_review_to_words(review))
    if(i % 1000 == 0):
        print("We are currently at review %d" % i)

# Create a vectorizer
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

# Fit_transform() 2 functions:
# 1. Fits the model and learns vocabulary
# 2. Transforms training data into feature vectors.
data_features = vectorizer.fit_transform(clean_reviews)
data_features = data_features.toarray()

# Initializing random forest classifier and run it
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(data_features, train["sentiment"])

#TODO: Test data