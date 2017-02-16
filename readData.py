import pandas as pd
from bs4 import BeautifulSoup as bs
import re

#read in the data
train = pd.read_csv("data\labeledTrainData.tsv", header=0, \
                    delimiter = "\t", quoting=3)

#remove tags from one movie review
noTag = bs(train["review"][0])

#remove punctuation
letter_only = re.sub("[^a-zA-Z]", " ", noTag.get_text())
print(train["review"][0])
print letter_only

