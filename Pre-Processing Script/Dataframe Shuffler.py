import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 100)

filename = "D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\Data To Label\\output_remaining.csv"

with open(filename, 'r', encoding='utf8', errors='ignore') as f:  # Open input file
    df = pd.read_csv(f, header=0)
    print(df.head())


    shuffled_df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    # print(shuffled_df.shape)
    print(shuffled_df.head())
    #
    shuffled_df.to_csv('ITER_2_output_remaining_shuffled.csv', encoding="utf8")
