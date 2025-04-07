from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def reduce_mem_usage(df):
    """
    Iterate through all the numeric columns of a dataframe and
    modify the data type to reduce memory usage.
    """

    print('\nTriggering memory optimization.......\n')

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print(
        'Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def print_pie(df: pd.DataFrame, users: List, color_dict, save=False) -> None:
    """
    Compare the proportions of genres rated by 2 users in a pie chart

    Input:
    df - (pd.DataFrame) Matrix containing:
            user_ids as index
            genres as columns
            number of ratings as values
    user - (List) List containing two user ids
    color_dict - (Dict) Color to use for each genre
    save - (bool) If True, save the pie chart
    """

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for i, user in enumerate(users):

        plt.subplot(1, 2, i + 1)

        df_summary = df.loc[df['user_id'] == user].genres.value_counts()

        # only print the % label if it's > 6%
        autopct = lambda pct: '{:1.0f}%'.format(pct) if pct > 6 else ''

        patches, texts, pct_texts = plt.pie(
            df_summary,
            labels=df_summary.index,
            startangle=90,
            counterclock=False,
            rotatelabels=True,
            autopct=autopct,
            colors=[color_dict[v] for v in df_summary.index],
        )

        # don't print the last 5 wedges as it might be small and overlapping
        end = df_summary.shape[0]
        start = end - 5
        for i in range(start, end):
            texts[i].set_visible(False)

        plt.axis('square')
        plt.setp(pct_texts, **{'weight': 'bold', 'fontsize': 10})
        plt.title('Genre distribution of User ID {}'.format(user), y=1.3)

    plt.suptitle('Compare Genre Distribution of two users', y=1.4)

    if save:
        fig.savefig(
            '../images/genre_distribution.png',
            transparent=True,
            pad_inches=0,
            bbox_inches='tight',
            facecolor='white',
            dpi=1200)

    plt.show()
