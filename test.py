from nltk import word_tokenize
from nltk.util import ngrams
from nltk import bigrams
import nltk
import re

def hasSleepKeywords(clean_words):  # Method returns true/false for presence of sleep keywords in clean text.
    list_sleep_words = ["bed", "sleep", "sack", "insomnia", "dodo", "zzz", "siesta", "tired", "nosleep",
                        "cantsleep", "exhausted", "sleepless", "awake", "late", "rest", "asleep",
                        "slept", "sleeping", "sleepy", "asleep", "nap", "oclock", "melatonin",
                        "ambien", "zolpidem", "lunesta", "intermezzo", "trazadone", "eszopiclone", "zaleplon"]
    list_sleep_bigrams = [["pass", "out"], ["get", "up"], ["wake", "up"],
                          ["sleep", "schedule"], ["hours", "sleep"], ["3", "hours"],
                          ["4", "hours"], ["5", "hours"], ["6", "hours"],
                          ["7", "hours"], ["8", "hours"], ["three", "hours"],
                          ["four", "hours"], ["five", "hours"], ["six", "hours"],
                          ["seven", "hours"], ["eight", "hours"], ["close", "eyes"],
                          ["at", "night"], ["the", "night"]]

    list_clean_words = re.split(' +', clean_words)

    for word in list_clean_words:
        if word in list_sleep_words:
            return True

    tweet_dict_bigrams = list(bigrams(list_clean_words))
    for each_bigram in tweet_dict_bigrams:
        for sleep_bigram in list_sleep_bigrams:
            if sleep_bigram[0] == each_bigram[0] and sleep_bigram[1] == each_bigram[1]:
                return True

    return False

test = "I only got 5 hours last night"
print(hasSleepKeywords(test))