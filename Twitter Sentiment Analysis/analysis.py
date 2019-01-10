#Importing the libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import nltk
from nltk.tokenize import word_tokenize     #For the word tokenization
from nltk.corpus import stopwords

style.use('ggplot')

def FrequencyDistribution(data,words,max_size=None):
    word_dict = dict()
    for word in words:
        word_dict[word] = 0
    
    for rev in data.split('\n'):
        rev = word_tokenize(rev)
        for word in rev:
            if word in words:
                word_dict[word] += 1
                
    word_dict = sorted(word_dict.items(),key=lambda t:t[1],reverse=True)
    if max_size == None:
        return list(word_dict.keys()) 
    else:
        req_words = list()
        for word in word_dict:
            if len(req_words) == max_size:
                break
            req_words.append(word[0])
    return req_words

#Preparing the Bag of Words Model
negative_reviews = open('Reviews/neg.txt','r').read()
positive_reviews = open('Reviews/pos.txt','r').read()

words = list()
data = list()       #In the form of (review , category)

stop_words = stopwords.words('english')
'''
The words which describe the review mainly fall in one of the following categories:
    Adverb , Adjective , Verb.
POS Tag: 
        JJ	adjective	'big'
        JJR	adjective, comparative	'bigger'
        JJS	adjective, superlative	'biggest'
        RB	adverb	very, silently,
        RBR	adverb, comparative	better
        RBS	adverb, superlative	best
        RP	particle	give up
        VB	verb, base form	take
        VBD	verb, past tense	took
        VBG	verb, gerund/present participle	taking
        VBN	verb, past participle	taken
        VBP	verb, sing. present, non-3d	take
        VBZ	verb, 3rd person sing. present	takes
So we will be including only those words which fall in the desired category. 
'''
allowed_types = ['J','R','V']

#Negative Reviews
for review in negative_reviews.split('\n'):
    rev_data = (review,'neg')
    rev_words = word_tokenize(review)
    tags = nltk.pos_tag(rev_words)
    
    data.append(rev_data)
    for word_tag in tags:
        if word_tag[1][0] in allowed_types:
            #Not in StopWords and not have been included before
            if word_tag[0].lower() not in stop_words and word_tag[0].lower() not in words:
                words.append(word_tag[0].lower())

#Positive Reviews
for review in positive_reviews.split('\n'):
    rev_data = (review,'pos')
    rev_words = word_tokenize(review)
    tags = nltk.pos_tag(rev_words)
    
    data.append(rev_data)
    for word_tag in tags:
        if word_tag[1][0] in allowed_types:
            #Not in StopWords and not have been included before
            if word_tag[0].lower() not in stop_words and word_tag[0].lower() not in words:
                words.append(word_tag[0].lower())

#Shuffling the data
import random
random.shuffle(data)

#Finding the most suited words
total_reviews = negative_reviews + '\n' + positive_reviews
words = FrequencyDistribution(total_reviews,words,max_size=8000)

#Feature Set
def find_feature(rev):
    label = rev[1]
    rev_data = rev[0]
    rev_data = word_tokenize(rev_data)
    feature = list()
    
    for w in words:
        f_val = (w in rev_data)
        feature.append(f_val)
    feature.append(label)
    return feature

featureset = np.array([find_feature(rev) for rev in data])
X = featureset[:,:-1]
y = featureset[:,-1]

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.06209,random_state=0)

