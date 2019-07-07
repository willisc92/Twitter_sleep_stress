import json
import pandas as pd


pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 100)



filename = "G:\\Users\\Willis\\Documents\\MENG Software\\ENSF 619-04 Machine Learning\\ensf-619-3-4-project\\iterative_output_all.json"
list_dicts_to_df = []

with open(filename, 'r', encoding='utf8', errors='ignore') as f:  # Open input file
    for line in f:
        clean_tweet_dict = json.loads(line)  # Load the .json object into a dictionary.
        list_dicts_to_df.append(clean_tweet_dict)

clean_tweet_df = pd.DataFrame(list_dicts_to_df)

print("Printing head of original dataframe")
print(clean_tweet_df.shape)
print(clean_tweet_df.head())

clean_tweet_df.set_index('index', inplace=True)


clean_tweet_df.rename(columns ={'place_country':'country',
                                'place_latitude':'latitude',
                                'place_longitude':'longitude',
                                'place_name':'city',
                                'place_province':'province',
                                'elapsed_time_in_days': 'account_life_days'
                                }, inplace=True)

reindex_order = ["label", "filtered_text", "orig_text", 'account_life_days', 'latitude', 'longitude',
                 'city', 'province', 'country', 'tweet_day', 'tweet_hour', 'tweet_weekday', 'tweet_year', 'user_ID', 'user_creation_day',
                 'user_creation_month', 'user_creation_year', 'user_followers_count', 'user_friends_count', 'user_listed_count',
                 'user_statuses_count', 'user_verified']

clean_tweet_df = clean_tweet_df[reindex_order]

def replace_commas_in_orig_text(row):
    return row["orig_text"].replace(",", " ")

def replace_commas_in_city(row):
    return row["city"].replace(",", " ")

def correctProvince(row):
    incorrect_provinces = ["Subd. C", "Toronto", "Subd. B", "Vancouver", "Subd. A", "Montréal", "Subd. D", "Calgary", "Subd. O", "Nouveau-Brunswick"]
    corrections = ["Newfoundland and Labrador", "Ontario", "Nova Scotia", "British Columbia", "Newfoundland and Labrador", "Québec", "Newfoundland and Labrador",
                   "Alberta", "Newfoundland and Labrador", "New Brunswick"]
    incorrect_correct_dict = dict(zip(incorrect_provinces, corrections))
    if row["province"] in incorrect_provinces:
        return incorrect_correct_dict[row["province"]]
    else:
        return row["province"]

clean_tweet_df["orig_text"] = clean_tweet_df.apply(replace_commas_in_orig_text, axis=1)
clean_tweet_df["city"] = clean_tweet_df.apply(replace_commas_in_city, axis=1)
clean_tweet_df["province"] = clean_tweet_df.apply(correctProvince, axis=1)

list_provinces = ["Manitoba", "Québec", "Yukon", "Nova Scotia", "Northwest Territories", "Newfoundland and Labrador",
                     "Alberta", "Ontario", "British Columbia", "New Brunswick", "Saskatchewan", "Prince Edward Island"]

in_list_provinces = clean_tweet_df["province"].isin(list_provinces)

filtered_pandas_df = clean_tweet_df[in_list_provinces]

shuffled_df = filtered_pandas_df.sample(frac=1, random_state=1).reset_index(drop=True)
print(shuffled_df.shape)
print(shuffled_df.head())

shuffled_df.to_csv('ready_to_label_from_local_iter_ALL.csv', encoding="utf8")
