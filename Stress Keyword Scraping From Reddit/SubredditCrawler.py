#! usr/bin/env python3

import praw
print (praw.__version__)
import pandas as pd

reddit = praw.Reddit(client_id='-BcE9jCFFAG2aw', \
                     client_secret='SECRET KEY GOES HERE', \
                     user_agent='<python>:<com.stresscrawler>:<v1.0> (by /u/Koolhaus)', \
                     username='Kooolhaus', \
                     password='PASSWORD GOES HERE')


topics_dict = {"title": [], "body": []}


for submission in reddit.subreddit('cantsleep').top(limit=1000):
    topics_dict["title"].append(submission.title)
    topics_dict["body"].append(submission.selftext)

topics_data = pd.DataFrame(topics_dict)
topics_data.to_csv('Scraped_cantsleep_Subreddit_data_top.csv', index=False)

print("Scraping complete.")