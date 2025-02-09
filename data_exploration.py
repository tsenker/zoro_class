# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:34:16 2025

@author: mxjar
"""

import pandas as pd
import numpy as np
import nltk
import re
import scipy
import sklearn
import spacy
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def remove_punctuation(text):
    return re.sub(r'[^\w\s]','',text)

def remove_numbers(text):
    return re.sub(r'[0-9]+','',text)

def whitespace_strip(text):
    return re.sub(r'\s+', ' ', text)



two_letter_words = ['AH', 'AM', 'AN','AS', 'AT', 'AY','BE','BY','DO','GO', 'HA', 'HE','IF', 'IN', 'IS', 'IT','LO', 'ME', 'MY', 'OF', 'OH', 'ON', 'OR', 'OX', 'SO','TO', 'UP', 'US','WE']
tlw = pd.DataFrame({'words':two_letter_words})
tlw['words'] = tlw['words'].str.lower()
tlw = set(tlw['words'])

zt = pd.read_csv('combined_zoro_texts.csv')
ct = pd.read_csv('chrs_texts.csv')

# some very basic data cleaning
zt['text'] = zt['text'].str.replace('translation: ','')
zt['text'] = zt['text'].str.replace('i.e.','')
zt = zt[zt['text'].str.len() > 50].reset_index(drop=True)

zt['rel'] = 'zoro'
ct['rel'] = 'chrs'

texts = pd.concat([zt,ct.sample(len(zt))],axis = 0).reset_index(drop=True).iloc[:,1:]
texts.columns = ['text','source','rel']

texts['text'] = texts['text'].str.split()

# removing nonsense 2-char fragments
stop_words = set(stopwords.words('english'))

texts['text'] = texts['text'].apply(lambda x: [i for i in x if (len(i)!=2  or (len(i)==2 and i in tlw)) and i not in stop_words])

# 2. Stop word removal

texts['text'] = texts['text'].apply(lambda x: ' '.join(x)).str.lower()
texts['text'] = texts['text'].apply(lambda x: whitespace_strip(remove_numbers(remove_punctuation(x))))



# let's lemmatize
def lemma_helper(text):
    doc = nlp(text)
    c = []
    for token in doc:
        c.append(token.lemma_)
    return ' '.join(c)

texts['text_l'] = texts['text'].apply(lambda x: lemma_helper(x))

def get_ngrams(text, n):
  """
  Extracts all n-grams from the given text.

  Args:
    text: The input text.
    n: The size of the n-grams (e.g., 2 for bigrams, 3 for trigrams).

  Returns:
    A list of n-grams.
  """
  doc = nlp(text)
  ngrams = list(zip(*[text.split()[i:] for i in range(n)]))
  return [" ".join(gram) for gram in ngrams]

texts['bigrams'] = texts['text'].apply(lambda x: get_ngrams(x,2))
texts['trigrams'] = texts['text'].apply(lambda x: get_ngrams(x,3))

### let's vectorize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion



tfidf = TfidfVectorizer(stop_words='english')

vectorizer = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    max_features=40000,
    )),
    
    ('char_tfidf', TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4),
    max_features=40000,
    ))
])


vectorizer.fit(texts['text_l'])
terms = vectorizer.get_feature_names_out()

vectorized_text = vectorizer.transform(texts['text_l'])

y = np.array(texts['rel'])

from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
vt_scaled = scaler.fit_transform(vectorized_text) 




from sklearn.decomposition import PCA

pca_init = PCA(n_components= 120)
#vt_pca = pca_init.fit(vt.T)
vt_pca_fit = pca_init.fit(vt_scaled)
vt_pca = pca_init.fit_transform(vt_scaled)
#vt_pca2 = PCA(n_components = 5).fit(vt.T)






cumulative_variance = np.cumsum(vt_pca_fit.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Different Principal Components')
plt.grid()
plt.show()







