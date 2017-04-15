import numpy as np
import pandas as pd
import re
from functools import reduce, partial
from functional import compose
from gensim.models.word2vec import Word2Vec
import pickle
from gensim.models.phrases import Phraser, Phrases
from random import sample
from sklearn.cluster import KMeans
from word_vectoriser import WordVectoriser


def make_text_cleaners():
    subs = ((r'https\w+$', ' '), (r'https://\w+$', ' '), ('http\w+\s', ' '), (r'@\w+', '_TN'),
           (r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', '_EM'), (r'\w+:\/\/\S+', r'_U'),
           (r' +', ' ',), (r'([^!\?])(\?{2,})(\Z|[^!\?])', r'\1 _BQ\n\3'), (r'([^\.])(\.{2,})', r'\1 _SS\n'),
           (r'([^!\?])(\?|!{2,})(\Z|[^!\?])', r'\1 _BX\n\3'), (r'([^!\?])\?(\Z|[^!\?])', r'\1 _Q\n\2'),
           (r'([^!\?])!(\Z|[^!\?])', r'\1 _X\n\2'), (r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL'),
           (r'(\w+)\.(\w+)', r'\1\2'), (r'[^a-zA-Z\s]', ''), (r'([#%&\*\$]{2,})(\w*)', r'\1\2 _SW'),
           (r' [8x;:=]-?(?:\)|\}|\]|>){2,}', r' _BS'), (r' (?:[;:=]-?[\)\}\]d>])|(?:<3)', r' _S'),
           (r' [x:=]-?(?:\(|\[|\||\\|/|\{|<){2,}', r' _BF'), (r' [x:=]-?(?:\(|\[|\||\\|/|\{|<){2,}', r' _BF'),
           (r' [x:=]-?[\(\[\|\\/\{<]', r' _F'), (r'\s\s', r' '))
    replaces = (('"', ' '), ('\'', ' '), ('_', ' '), ('-', ' '), ('\n', ' '), ('\\n', ' '), ('\'', ' '), ('RT', ''))
    def repl(original, new, word):
        return word.replace(original, new)
    substitute = reduce(compose, [partial(re.sub, x, y) for (x, y) in subs])
    replace = reduce(compose, [partial(repl, x, y) for (x, y) in replaces])
    return compose(substitute, replace)


def standardise(vector):
    """takes a (av, 100) vector and returns a (12,) vector"""
    if len(vector) == 0:
        return np.zeros(12)
    reduced = np.array([np.mean(x) for x in vector])
    if len(reduced) < 12:
        missing = 12 - len(reduced)
        padding = np.mean(reduced)
        return np.append(reduced, np.array([padding for _ in range(missing)]))
    else:
        return np.array(sample(list(reduced), 12))


def build_w2v(df):
    try:
        with open("w2vpickle.pkl", "rb") as f:
            w2v = pickle.load(f)
            print("model loaded")
    except FileNotFoundError as e:
        print(e)
        w2v = Word2Vec(df['phrased_sentences'].values, min_count=5)
        print("model created")

        with open("w2vpickle.pkl", "wb") as f:
            pickle.dump(w2v, f)
    return w2v


def main(filename):
    with open(filename, 'r') as f:
        raw_df = pd.read_csv(f, header=0, names=['Text']).dropna(axis=0).drop_duplicates()
    raw_df = raw_df.reset_index(drop=True)
    df = raw_df.copy()

    cleaner = make_text_cleaners()
    df['Clean Text'] = df['Text'].apply(cleaner)
    df['Clean Text'] = df['Clean Text'].apply(lambda x: x.split())

    """phrase pre-processing"""
    sents = df['Clean Text'].values
    phrases = Phrases(sents, min_count=5, threshold=100)
    bigramphraser = Phraser(phrases)
    """produce a representation of the text including 2 and 3 word phrases"""
    trg_phrases = Phrases(bigramphraser[sents], min_count=5, threshold=100)
    trigramphraser = Phraser(trg_phrases)
    df['Phrased Sentences'] = list(trigramphraser[list(bigramphraser[sents])])

    df = df.drop(['Clean Text'], axis=1)

    w2v = build_w2v(df)

    vectoriser = WordVectoriser(w2v.wv).fit(df['Phrased Sentences'].values)
    vecs = np.array([x for x in
                    [vectoriser.transform(s) for s in
                     df['Phrased Sentences'].values]])

    print("transformed {} tweets into {} vectors of shape (av, {}) where av = {} \n"
              .format(len(df['Phrased Sentences']),
                      vecs.shape[0],
                      vecs[0].shape[1],
                      np.mean([v.shape[0] for v in vecs])))

    vecs_std = np.array([standardise(v) for v in vecs])

    kmeans = KMeans(n_clusters=3, random_state=0).fit(vecs_std)
    df['cluster'] = kmeans.labels_

    with open("_".join([filename, "clstr", ".csv"]), "w") as f:
        df.to_csv(f)

if __name__ == "__main__":
    pass