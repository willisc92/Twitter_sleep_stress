import pandas as pd
import glob

path = r'D:\Documents\MEng Software\ENSF 619-4 Machine Learning\ensf-619-3-4-project\Sleep Keyword Scraping From Reddit'
all_files = glob.glob(path + "/*.csv")

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 100)

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col = None, header = 0)
    li.append(df)

frame = pd.concat(li, axis = 0, ignore_index = True)
print(frame.shape)

frame.drop_duplicates(inplace = True)

print(frame.shape)
frame["title_and_body"] = frame["title"] + " " + frame["body"]
frame = frame["title_and_body"]

frame.to_csv("compiled_sleep_subreddit_titles_and_bodies.csv")
print(frame.head(20))
