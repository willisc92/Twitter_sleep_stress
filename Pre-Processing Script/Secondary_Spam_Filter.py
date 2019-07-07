import pandas as pd
from nltk import bigrams
import nltk
from nltk import ngrams

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 100)

filename = "ITER_2_labeled_combined.csv"

def filter_spam(row):
    spam_bigrams = [["job", "opening"], ["current", "weather"], ["job", "openings"], ["apply", "for"],
                    ["good","job"],["great","job"], ["now","open"]]
    spam_trigrams = [["land", "a", "job"]]
    tokenize = nltk.word_tokenize(row["filtered_text"])
    tweet_text_bigrams = ngrams(tokenize, 2)

    for each_bigram in tweet_text_bigrams:
        for spam_bigram in spam_bigrams:
            if (each_bigram[0] == spam_bigram[0] and each_bigram[1] == spam_bigram[1]):
                return("Spam")

    tweet_text_trigrams = ngrams(tokenize, 3)
    for each_trigram in tweet_text_trigrams:
        for spam_trigram in spam_trigrams:
            if (each_trigram[0] ==  spam_trigram[0] and each_trigram[1] == spam_trigram[1] and each_trigram[2] == spam_trigram[2]):
                return("Spam")


    return row["label"]

def province_correction(row):
    if row["province"].startswith("Q"):
        return "Quebec"
    else:
        return row["province"]

with open(filename, 'r', encoding='utf8', errors='ignore') as f:  # Open input file
    df = pd.read_csv(f, header=0)
    # print(df.head())
    # print(df.shape)
    df["label"] = df.apply(filter_spam, axis=1)
    df["province"] = df.apply(province_correction, axis=1)
    no_spam = df["label"] != "Spam"
    df = df[no_spam].reset_index(drop=True)
    df.to_csv('ITER_2_labeled_combined_no_spam.csv', encoding="utf8")



