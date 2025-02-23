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

texts['text'] = texts['text'].str.lower().str.split()

# removing nonsense 2-char fragments
stop_words = set(stopwords.words('english'))

texts['text'] = texts['text'].apply(lambda x: [i for i in x if (len(i)!=2  or (len(i)==2 and i in tlw)) and i not in stop_words])

# 2. Stop word removal

texts['text'] = texts['text'].apply(lambda x: ' '.join(x)).str.lower()
texts['text'] = texts['text'].apply(lambda x: whitespace_strip(remove_numbers(remove_punctuation(x))))


# let's figure out what the most specific religious words are
sw = texts['text'].str.split()
word_freq = [i for j in sw for i in j]
word_freq = pd.DataFrame({'wf':word_freq,}).groupby(['wf']).size().reset_index()
word_freq.columns = ['word','count']
word_freq = word_freq.sort_values(by = 'count', ascending = False).reset_index(drop = True)

from collections import defaultdict

word_freq = defaultdict(lambda: defaultdict(int))

for text,rel in zip(texts['text'], texts['rel']):
    for word in text.lower().split():
        word_freq[rel][word] += 1

word_freq = pd.DataFrame(word_freq).reset_index().fillna(0)
word_freq.columns = ['word','zoro','chrs']
word_freq['ratio'] = word_freq['zoro']/word_freq['chrs']
word_freq['ratio'] = np.where(word_freq['ratio'] == np.inf,99999,word_freq['ratio'])

religious_figures = ['mazda',
                     'ahura',
                     'yamaha',
                     'zarathustra',
                     'fra',
                     'vas',
                     'dallas',
                     'hormazd',
                     'pittman',
                     'frosh',
                     'zat',
                     'ashishvangh',
                     'audi',
                     'haoma',
                     'spenta',
                     'amesha',
                     'yacht',
                     'fravashi',
                     'wizard',
                     'ameshaspands',
                     'jamshid',
                     'chordal',
                     'jesus',
                     'christ',
                     'paul',
                     'pharisee',
                     'gentile',
                     'paul',
                     'simon',
                     'pilate',
                     'david',
                     'mary',
                     'herod',
                     'phillip',
                     'judas',
                     'joseph',
                     'isaiah',
                     'elijah',
                     'jacob',
                     'peter',
                     'abraham',
                     'james',
                     'disciple',
                     'creator',
                     'immortal',
                     'meter',
                     'protector']

religious_values = ['religion',
                    'rectitude',
                    'powerful',
                    'beneficient',
                    'happiness',
                    'prosperity',
                    'amongst',
                    'victorious',
                    'wide',
                    'data',
                    'glorious',
                    'bountiful',
                    'protector',
                    'excellence',
                    'boon',
                    'nonchalance',
                    'corporeal',
                    'heroic',
                    'health',
                    'divine',
                    'smite',
                    'undefile',
                    'religious',
                    'perfection',
                    'swift',
                    'famous',
                    'adoration',
                    'bliss',
                    'virtuous',
                    'aright',
                    'friendship',
                    'mindedness',
                    'propitiation',
                    'brotherhood',
                    'protection',
                    'intelligence',
                    'real',
                    'existence',
                    'greatness',
                    'efficacious',
                    'omniscient',
                    'hatred',
                    'cruel',
                    'vigour',
                    'brilliant',
                    'prosperous',
                    'glorification',
                    'triumphant',
                    'invocation',
                    'path',
                    'best',
                    'sacred',
                    'courage',
                    'goodness',
                    'wicked',
                    'humility',
                    'conscience',
                    'excellent',
                    'wealth',
                    'worth',
                    'pure',
                    'unclean',
                    'command',
                    'believe',
                    'drink',
                    'burn',
                    'know',
                    'begot',
                    'baptize',
                    'crucify',
                    'lawful',
                    'besought',
                    'write',
                    'cry',
                    'depart',
                    'die',
                    'fall',
                    'arise']

specified_places_nationalities = ['iranian',
                                  'jew',
                                  'israel',
                                  'church',
                                  'synagogue',
                                  'jerusalem',
                                  'egypt',
                                  'judaea',
                                  'wilderness',
                                  'country',
                                  'pasture',
                                  'universe']


# let's lemmatize
def lemma_helper(text):
    doc = nlp(text)
    c = []
    for token in doc:
        c.append(token.lemma_)
    return ' '.join(c)

texts['text_l'] = texts['text'].apply(lambda x: lemma_helper(x))

def replace_words(sentence):
    words = sentence.split()
    new_words = ['religious_figure' if word.lower() in set(religious_figures) else word for word in words]
    new_words = ['religious_value' if word.lower() in set(religious_values) else word for word in new_words]     
    new_words = ['place_or_nationality' if word.lower() in set(specified_places_nationalities) else word for word in new_words]          
    return ' '.join(new_words)



texts['text_l'] = texts['text_l'].apply(lambda x: replace_words(x))



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

vectorizer_old = FeatureUnion([
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

vectorizer = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
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
from sklearn.decomposition import TruncatedSVD


svd_init = TruncatedSVD(n_components = 50)

vt_svd_fit = svd_init.fit(vt_scaled)
vt_svd = svd_init.fit_transform(vt_scaled)


# Get feature names (words)
feature_names = np.array(vectorizer.get_feature_names_out())

# Find top words influencing each principal component
for i, component in enumerate(svd_init.components_):
    top_words = feature_names[np.argsort(-np.abs(component))[:5]]
    print(f"Principal Component {i+1}: {top_words}")


cumulative_variance = np.cumsum(vt_svd_fit.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Different Principal Components')
plt.grid()
plt.show()



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

vt_svd_df = pd.DataFrame(vt_svd)
vt_svd_df.columns = vt_svd_df.columns.astype(str)


x_test, x_train, y_test, y_train = train_test_split(vt_svd_df,y,test_size= 0.7)

model_log = LogisticRegression()
model_log.fit(x_train, y_train)

y_prob_log = model_log.predict_proba(x_test)

log_predict = pd.DataFrame({'predictions':model_log.predict(x_test),'chrs_prob':y_prob_log[:,0],'zoro_prob':y_prob_log[:,1]})
log_predict.index = x_test.index

accuracy_log = accuracy_score(y_test, log_predict['predictions'])
precision_log, recall_log, f1_log, support_log = precision_recall_fscore_support(y_test, log_predict['predictions'], average=None)

print('Accuracy is '+str(accuracy_log))
print('Precision is '+str(precision_log))
print('Recall is '+str(recall_log))
print('F1 is '+str(f1_log))
print('Support is '+str(support_log))

log_examination_df = pd.DataFrame({'text':texts[texts.index.isin(set(x_test.index))==True]['text'],
                               'rel':texts[texts.index.isin(set(x_test.index))==True]['rel']})

log_examination_df = log_examination_df.merge(log_predict, left_index = True, right_index=True)


plt.figure(figsize=(8, 6))
plt.hist(log_examination_df['zoro_prob'])
plt.xlabel('Zoroastrian Likelihood')
plt.ylabel('Number of instances')
plt.title('Explained Variance by Different Principal Components')
plt.grid()
plt.show()


vt_svd_df['rel'] = texts['rel']
vt_svd_df['rel_bin'] = np.where(vt_svd_df['rel'] == 'chrs',1,0)

colors = ['red', 'blue']
group_colors = [colors[g] for g in vt_svd_df['rel_bin']]


plt.figure(figsize=(8, 6))
plt.scatter(vt_svd_df.iloc[:,0],vt_svd_df.iloc[:,1], c=group_colors)
plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')
plt.title('Zoroastrian Christian text comparison: most prominent SVD components visualized')
plt.grid()
plt.show()



from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

mm_scaler = MinMaxScaler()
x_train_mm_scaled = mm_scaler.fit_transform(x_train)
x_test_mm_scaled = mm_scaler.fit_transform(x_test)

model_nb = MultinomialNB()
model_nb.fit(x_train_mm_scaled, y_train)

y_prob_nb = model_nb.predict_proba(x_test_mm_scaled)

nb_predict = pd.DataFrame({'predictions':model_nb.predict(x_test),'chrs_prob':y_prob_nb[:,0],'zoro_prob':y_prob_nb[:,1]})
nb_predict.index = x_test.index

accuracy_nb = accuracy_score(y_test, nb_predict['predictions'])
precision_nb, recall_nb, f1_nb, support_nb = precision_recall_fscore_support(y_test, nb_predict['predictions'], average=None)

print('Accuracy is '+str(accuracy_nb))
print('Precision is '+str(precision_nb))
print('Recall is '+str(recall_nb))
print('F1 is '+str(f1_nb))
print('Support is '+str(support_nb))

nb_examination_df = pd.DataFrame({'text':texts[texts.index.isin(set(x_test.index))==True]['text'],
                               'rel':texts[texts.index.isin(set(x_test.index))==True]['rel']})

nb_examination_df = nb_examination_df.merge(nb_predict, left_index = True, right_index=True)

from sklearn.svm import SVC

model_svm = SVC(probability = True)
model_svm.fit(x_train, y_train)

y_prob_svm = model_svm.predict_proba(x_test)

svm_predict = pd.DataFrame({'predictions':model_svm.predict(x_test),'chrs_prob':y_prob_svm[:,0],'zoro_prob':y_prob_svm[:,1]})
svm_predict.index = x_test.index

accuracy_svm = accuracy_score(y_test, svm_predict['predictions'])
precision_svm, recall_svm, f1_svm, support_svm = precision_recall_fscore_support(y_test, svm_predict['predictions'], average=None)

print('Accuracy is '+str(accuracy_svm))
print('Precision is '+str(precision_svm))
print('Recall is '+str(recall_svm))
print('F1 is '+str(f1_svm))
print('Support is '+str(support_svm))

examination_df_svm = pd.DataFrame({'text':texts[texts.index.isin(set(x_test.index))==True]['text'],
                               'rel':texts[texts.index.isin(set(x_test.index))==True]['rel']})

examination_df_svm = examination_df_svm.merge(svm_predict, left_index = True, right_index=True)


# now let's put it all together

nb_examination_df['miss'] = np.where(nb_examination_df['rel'] != nb_examination_df['predictions'],1,0)
log_examination_df['miss'] = np.where(log_examination_df['rel'] != log_examination_df['predictions'],1,0)
examination_df_svm['miss'] = np.where(examination_df_svm['rel'] != examination_df_svm['predictions'],1,0)

union_misclass_idx = set(log_examination_df[log_examination_df['miss'] == 1].index) & set(nb_examination_df[nb_examination_df['miss'] == 1].index) & set(examination_df_svm[examination_df_svm['miss'] == 1].index)

common_misclass_sentences = pd.DataFrame({'text':texts[texts.index.isin(union_misclass_idx)==True]['text'],
                                          'text_anon':texts[texts.index.isin(union_misclass_idx)==True]['text_l'],
                                          'rel':texts[texts.index.isin(union_misclass_idx)==True]['rel'],
                                          'prediction':log_examination_df[log_examination_df.index.isin(union_misclass_idx)==True]['predictions'],
                                          'log_zoro_prob':log_examination_df[log_examination_df.index.isin(union_misclass_idx)==True]['zoro_prob'],
                                          'log_chrs_prob':log_examination_df[log_examination_df.index.isin(union_misclass_idx)==True]['chrs_prob'],
                                          'nb_zoro_prob':nb_examination_df[nb_examination_df.index.isin(union_misclass_idx)==True]['zoro_prob'],
                                          'nb_chrs_prob':nb_examination_df[nb_examination_df.index.isin(union_misclass_idx)==True]['chrs_prob'],
                                          'svm_zoro_prob':examination_df_svm[examination_df_svm.index.isin(union_misclass_idx)==True]['zoro_prob'],
                                          'svm_chrs_prob':examination_df_svm[examination_df_svm.index.isin(union_misclass_idx)==True]['chrs_prob'],})

# let's analyze the SVD decomposition qualitatively
quantiles_used = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
c1_range = np.quantile(vt_svd_df.iloc[:,0],quantiles_used)
c2_range = np.quantile(vt_svd_df.iloc[:,1],quantiles_used)
c3_range = np.quantile(vt_svd_df.iloc[:,2],quantiles_used)
c4_range = np.quantile(vt_svd_df.iloc[:,3],quantiles_used)
c5_range = np.quantile(vt_svd_df.iloc[:,4],quantiles_used)



components = [1,2,3,4,5]

for e in components:
    print('THIS IS SVD COMPONENT '+str(e))
    print('')
    print('')
    for i,j in zip(np.quantile(vt_svd_df.iloc[:,e - 1],quantiles_used),quantiles_used):
        print('This is the '+str(100*j)+'th percentile text for this SVD component:')
        print(texts[texts.index == np.argmin(abs(vt_svd_df.iloc[:,e - 1] - i))].reset_index(drop=True)['text'][0])
        print('-------')
    print('-------------------')


