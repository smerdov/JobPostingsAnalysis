import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, DBSCAN
import tqdm
import os
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemma = WordNetLemmatizer()

vocabulary = ['skill', 'requirement', 'familiarity', 'proficiency', 'experience',
              'preferred', 'knowledge', 'ability', 'demonstrate']
vocabulary = set([stemmer.stem(word) for word in vocabulary])
my_stopwords = ['data', 'work', 'learn', 'year', 'use', 'strong', 'etc', 'field']
my_stopwords = set([stemmer.stem(word) for word in my_stopwords])


def filter_tokens(tokens):
    tokens_new = set()
    for token in tokens:
        if token in stopwords.words('english'):
            continue

        token = stemmer.stem(token)
        if token in stopwords.words('english'):
            continue

        if not token.isalpha():
            continue

        if token in my_stopwords:
            continue

        tokens_new.add(token)

    return tokens_new


def generate_count_matrix(df):
    path2save = 'count_matrix'
    if os.path.exists(path2save):
        df_skills = joblib.load(path2save)  # df_skills a.k.a. count_matrix
        return df_skills

    features_dict_list = []
    for (index, row) in tqdm.tqdm(df[:].iterrows()):
        description = row['job_description']

        sent_tokens = sent_tokenize(description)
        # assert len(sent_tokens) > 1

        description_sent = []
        for sent in sent_tokens:
            word_tokens = word_tokenize(sent)
            word_tokens_normalized = filter_tokens(word_tokens)
            description_sent.append(word_tokens_normalized)

        desc_counter = Counter()
        for sent in description_sent:
            if len(sent.intersection(vocabulary)) > 0:
                desc_counter.update(sent - vocabulary)

        features_dict_list.append(desc_counter)

    df_skills = pd.DataFrame(features_dict_list)
    joblib.dump(df_skills, path2save)
    return df_skills


def generate_reweighted_count_matrix(df_skills):
    top_skills_df = df_skills.sum().sort_values(ascending=False)[:500]
    top_skills = top_skills_df.index

    df_skills = df_skills.loc[:, top_skills]
    df_skills.fillna(0, inplace=True)

    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(df_skills).toarray()

    return tfidf_matrix


def clusterize(tfidf_matrix, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_prediction = kmeans.fit_predict(tfidf_matrix)
    return cluster_prediction


def interpret_results(cluster_prediction, df_skills, top_keywords=10):
    keywords4clusters = []
    clusters = np.unique(cluster_prediction)

    for cluster in clusters:
        df4cluster = df_skills.loc[cluster_prediction == cluster, :]
        top_skills4cluster = list(df4cluster.sum().sort_values(ascending=False)[:top_keywords].index)
        keywords4clusters.append(top_skills4cluster)

    return keywords4clusters
