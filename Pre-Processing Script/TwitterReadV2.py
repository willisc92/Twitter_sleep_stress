import json
import emoji
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder
import pytz
import re
import ssl
import certifi
import geopy.geocoders
from nltk import bigrams
import sys
import time
import os

ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx


class FilteredTweet:  # Class to store the clean Tweet data.
    def __init__(self, tweet_dict, list_sleep_words, list_sleep_bigrams, list_stress_words):
        self.index = None
        localtime = parseDate(tweet_dict)
        self.tweet_weekday = localtime.weekday()
        self.tweet_day = localtime.day
        self.tweet_hour = localtime.hour
        self.tweet_year = localtime.year
        self.orig_text = tweet_dict['text']
        self.filtered_text = parseTweetText(tweet_dict, list_sleep_words, list_sleep_bigrams, list_stress_words)

        self.user_ID = tweet_dict['user']['id']
        self.user_verified = tweet_dict['user']['verified']
        self.user_followers_count = tweet_dict['user']['followers_count']
        self.user_friends_count = tweet_dict['user']['friends_count']
        self.user_listed_count = tweet_dict['user']['listed_count']
        self.user_statuses_count = tweet_dict['user']['statuses_count']

        user_creation_time = datetime.strptime(tweet_dict['user']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        self.user_creation_year = user_creation_time.year
        self.user_creation_month = user_creation_time.month
        self.user_creation_day = user_creation_time.day

        self.place_name = tweet_dict['place']['name']

        city_and_province = tweet_dict['place']['full_name'].split(",")
        if len(city_and_province) > 1:
            self.place_province = city_and_province[1].strip()
        else:
            self.place_province = None

        self.place_country = tweet_dict['place']['country']
        self.place_longitude = float(tweet_dict['place']['bounding_box']['coordinates'][0][0][0])
        self.place_latitude = float(tweet_dict['place']['bounding_box']['coordinates'][0][0][1])

        tweet_date = datetime.strptime(tweet_dict["created_at"], '%a %b %d %H:%M:%S +0000 %Y')
        elapsed_time = tweet_date - user_creation_time
        self.elapsed_time_in_days = elapsed_time / timedelta(minutes=1) / 60 / 24

        self.label = None

    def __str__(self):  # Represent the tweet as a dictionary.
        return str(self.__dict__)

    def setIndex(
            self, the_index):  # Method will set the index.  Called when new clean tweet is written.
        self.index = the_index


def filterCountry(tweet_dict, country):  # Filter tweet object based on country.  Takes arguments of a standard tweet
    # dictionary and country (as string).
    if 'place' in tweet_dict.keys():
        if tweet_dict['place'] is not None:
            if 'country' in tweet_dict['place'].keys():
                if tweet_dict['place']['country'] == country:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False


def filterTrackable(tweet_dict):  # Filter tweet based on whether it is trackable or not.
    if parseTimeZone(tweet_dict) is not None:
        return True
    else:
        return False


def filterLanguage(tweet_dict, language):  # Filter tweet object based on language.
    if 'lang' in tweet_dict.keys():
        if tweet_dict['lang'] is not None:
            if tweet_dict['lang'] == language:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def filterOriginal(tweet_dict):  # Method to filter out tweets that are quotes or re-tweets.
    if 'is_quote_status' in tweet_dict.keys():
        if tweet_dict['is_quote_status']:
            return False
    if tweet_dict['retweeted']:
        return False
    else:
        return True


def parseDate(tweet_dict):  # Method to parse the date/time of a tweet into the tweeter's local time.
    date_info = tweet_dict['created_at']
    orig_date = datetime.strptime(date_info, '%a %b %d %H:%M:%S +0000 %Y')
    fmt = '%a, %b %d %Y %H:%M:%S'
    new_date = datetime.strptime(datetime.strftime(orig_date, fmt), fmt)
    tz = parseTimeZone(tweet_dict)
    localized_time = new_date.astimezone(pytz.timezone(tz))
    offset = int(str(localized_time)[-6:-3])
    delta = timedelta(hours=offset)
    localized_time = new_date + delta
    return localized_time


def parseTimeZone(tweet_dict):  # Method to retrieve the timezone based on the user coordinates.
    tf = TimezoneFinder()
    if 'place' in tweet_dict.keys():
        if tweet_dict['place'] is not None:
            if 'bounding_box' in tweet_dict['place'].keys():
                if 'coordinates' in tweet_dict['place']['bounding_box'].keys():
                    if len(tweet_dict['place']['bounding_box']['coordinates']) > 0:
                        longitude = float(tweet_dict['place']['bounding_box']['coordinates'][0][0][0])
                        latitude = float(tweet_dict['place']['bounding_box']['coordinates'][0][0][1])
                        timezone = tf.timezone_at(lat=latitude, lng=longitude)
                        if timezone is None:
                            timezone = tf.closest_timezone_at(lat=latitude, lng=longitude)
                        return timezone
                    else:
                        return None
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None


def filterTweet(tweet_dict, language, country):  # Apply language, original, country, and trackable filters to a tweet.
    return (filterCountry(tweet_dict, country) and filterLanguage(tweet_dict, language)
            and filterOriginal(tweet_dict) and filterTrackable(tweet_dict))


def is_emoji(s):  # Method to check if a string is an emoji.
    return s in emoji.UNICODE_EMOJI


def add_space(text):  # Method to add a space between word-emoji pairs.
    result = ''
    for char in text:
        if is_emoji(char):
            char = 'emoji_' + char + " "
            result += ' '
        result += char
    return result.strip()


def parseTweetText(tweetDict, list_sleep_words, list_sleep_bigrams,
                   list_stress_words):  # Method to demojize tweet text, URLs, remove punctuation, and move to lowercase.
    tweet_text = tweetDict["text"]
    tweet_text = tweet_text + " ."
    tweet_text = tweet_text.replace('&amp;', " and ")
    tweet_text = tweet_text.replace('+', " ")
    tweet_text = tweet_text.replace('=', " ")
    tweet_text = tweet_text.replace('\n', " ")
    tweet_text = tweet_text.replace('@', " AT_")
    tweet_text = tweet_text.replace('#', " ")
    tweet_text = tweet_text.replace('-', " ")
    tweet_text = tweet_text.replace('\'', "")
    tweet_text = add_space(tweet_text)
    tweet_text = emoji.demojize(tweet_text)

    filtered_words_1 = []
    clean_words_1 = []

    for word in re.split(' +', tweet_text):
        if word.startswith('https'):
            continue
        else:
            word = word.lower()
            word = word.replace("\\", " and ")
            word = word.replace("/", " and ")
            word = re.sub(r'[^a-z0-9\s_]', ' ', word)
            word = word.strip()
            clean_words_1.append(word)
    clean_sentence_1 = ' '.join(clean_words_1)
    clean_words_2 = re.split(' +', clean_sentence_1)
    for word in clean_words_2:
        while (word.startswith(" ") or word.endswith(" ")):
            word = word.strip()
        while (word.startswith("_") or word.endswith("_")):
            word = word.strip("_")

    if (checkSleepWords(clean_words_2, list_sleep_words, list_sleep_bigrams) or checkStressWords(clean_words_2,
                                                                                                 list_stress_words, list_stress_bigrams)):
        for word in re.split(' +', tweet_text):
            if word.startswith('https'):
                continue
            else:
                word = word.lower()
                word = word.replace("\\", " and ")
                word = word.replace("\/", " and ")
                word = re.sub(r'[^a-z0-9\s.,:;!?_]', '', word)
                word = word.strip()
                filtered_words_1.append(word)
        filtered_sentence_1 = ' '.join(filtered_words_1)
        filtered_words_2 = []
        filtered_sentence_2 = negation_sub(filtered_sentence_1)
        for word in re.split(' +', filtered_sentence_2):
            word = re.sub(r'[.,:;!?]', ' ', word)
            while (word.startswith(" ") or word.endswith(" ")):
                word = word.strip()
            while (word.startswith("_") or word.endswith("_")):
                word = word.strip("_")
            if (word != ""):
                filtered_words_2.append(word)
        last_filtered = ' '.join(filtered_words_2)
        last_filtered = last_filtered.replace("emoji_ ", "emoji_")
        return last_filtered
    else:
        return None


def negation_sub(text):
    transformed = re.sub(r'\b(?:not|no|never|aint|doesnt|havent|lacks|none|mightnt|shouldnt|'
                         r'cannot|dont|neither|nor|mustnt|wasnt|cant|hadnt|isnt|neednt|without|'
                         r'darent|hardly|lack|nothing|oughtnt|wouldnt|didnt|hasnt|lacking|nobody|'
                         r'nowhere|shant)\b[\w\s]+[.,:;!?]',
                         lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)),
                         text,
                         flags=re.IGNORECASE)
    return transformed


def checkSleepWords(list_words_to_check, sleep_words,
                    sleep_bigrams):  # Method to check for existence of sleep keywords/bigrams in filtered text.
    # Takes a list of words to check, list of sleep words, and list of sleep bigrams to compare to.
    for word in list_words_to_check:
        if word in sleep_words:
            return True
    
	tweet_dict_bigrams = list(bigrams(list_words_to_check))
    for each_bigram in tweet_dict_bigrams:
        for sleep_bigram in sleep_bigrams:
            if sleep_bigram[0] == each_bigram[0] and sleep_bigram[1] == each_bigram[0]:
                return True
    return False


def checkStressWords(list_words_to_check,
                     stress_words, stress_bigrams):  # Method to check for existence of stress keywords in filtered text.
    # Takes a list of words to check and a list of stress words to compare to.
    for word in list_words_to_check:
        if word in stress_words:
            return True

    tweet_dict_bigrams = list(bigrams(list_words_to_check))
    for each_bigram in tweet_dict_bigrams:
        for stress_bigram in stress_bigrams:
            if stress_bigram[0] == each_bigram[0] and stress_bigram[1] == each_bigram[0]:
                return True
    return False


list_sleep_words = ["sleep", "sack", "insomnia", "dodo", "zzz", "siesta", "tired", "nosleep",
                    "cantsleep", "rest", "asleep", "slept", "sleeping", "sleepy",
                    "ambien", "zolpidem", "lunesta", "intermezzo", "trazadone", "eszopiclone",
                    "zaleplon"]  # List of sleep words to check for.
list_sleep_bigrams = [["pass", "out"], ["get", "up"], ["wake", "up"],
                      ["power", "nap"], ["to", "bed"]]  # List of sleep bigrams to compare tweet text to.
list_stress_words = ["feel", "time", "life", "stress","school", "depression", "fucking", "sick", "stressor",
"anxiety","pressure","depressed","study","heart","pain","stressful"]  # List of stress keywords to compare tweet text to.
list_stress_bigrams = [["high", "school"], ["dont", "feel"], ["years", "ago"], ["dont", "care"], ["long", "time"],
["mental", "health"], ["feel", "bad"], ["suicidal", "thoughts"], ["feel", "guilty"], ["hard", "time"], ["mental", "illness"]]

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


start_time = time.time()

sys.stdout = Logger("iterative_output_log_all.txt")
index_clean = 1
num_eval = 0 
path_to_json = "H:\\Machine Learning Data\\New\\"
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
output_filename = "iterative_output_all.json"  # Name of clean .json file to write to.  Will have 1 clean tweet per line of file.

print("Beginning processing...")
for js in json_files:
    with open(os.path.join(path_to_json, js), 'r', encoding='utf8', errors='ignore') as f:  # Open input file
        with open(output_filename, 'a', encoding='utf8') as outputFile:  # Open output file
            for (i, line) in enumerate(f, 1):  # Iterate through each line in input file.
                tweet_dict = json.loads(line)  # Load the .json object into a dictionary.
                num_eval += 1
                try:
                    if filterTweet(tweet_dict, "en", "Canada"):  # Filter the tweet.
                        new_tweet = FilteredTweet(tweet_dict, list_sleep_words, list_sleep_bigrams,
                                                  list_stress_words)  # Create a new filtered tweet.
                        if new_tweet.filtered_text is None:
                            continue
                        else:
                            new_tweet.setIndex(index_clean)  # Set the index.
                            new_tweet_json = json.dumps(
                                new_tweet.__dict__)  # Create a new json object from the clean tweet.
                            clean_tweet_json = json.dumps(new_tweet.__dict__, indent=4)
                            print(clean_tweet_json)
                            outputFile.write(new_tweet_json)  # Write the new json object to the output file.
                            outputFile.write('\n')  # Write a newline to separate tweets in output file.
                            print("Current tweets written from file:", index_clean, ", Total Tweets evaluated from file:",
                                  i, ", Total Tweets read overall:", num_eval)
                            index_clean += 1
                except:
                    print("Encountered error in line", i, ": skipping this line.")
                    continue

print("****************************************")
print("Total clean tweets written from file: ", index_clean - 1)
print("Total tweets evaluated from file: ", num_eval)
end_time = time.time()
print("Time of execution in seconds:", end_time - start_time)