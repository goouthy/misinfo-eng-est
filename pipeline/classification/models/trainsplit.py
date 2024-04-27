import pandas as pd
from sklearn.model_selection import train_test_split

def splitandbalance(input, splitsize=0.2):
    train_df, val_df = train_test_split(input, test_size=splitsize, random_state=0) 

    minority_class = train_df['Label'].value_counts().idxmin()
    df_majority = train_df[train_df['Label'] != minority_class]
    df_minority = train_df[train_df['Label'] == minority_class]
    df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=0)
    upsampled_train_df = pd.concat([df_majority, df_minority_upsampled])

    return upsampled_train_df, val_df