"""
This script cleans movie data to support
my movie recommendation webapp.

IMPORTANT run from the main project root using the command
`python -m src.preprocessing_data_cleaning`
"""

import argparse
import logging
from typing import List
import os

import pandas as pd

from src import (path_imdb_clean, path_imdb_raw, path_links_clean,
                 path_links_raw, path_log_wrangling, path_ratings_clean,
                 path_ratings_raw, path_tags_clean, path_tags_raw,
                 reduce_mem_usage, folder_clean, path_imdb_raw_extra)

# activate logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=path_log_wrangling,
    format='%(asctime)s %(levelname)-8s %(message)s',
    filemode='a',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def clean_imdb(
        path_raw: str
) -> pd.DataFrame:
    """
    Load and clean raw IMDB/TMDB data

    INPUT:
    path_raw: raw folder where IMDB/TMDB data are stored in csv format

    OUTPUT:
    df: cleaned IMDB/TMDB dataset
    """

    logger.info('Cleaning TMDB/IMDB movie content data...')
    print('Cleaning TMDB/IMDB movie content data...')

    # load data
    df = pd.read_csv(path_raw, parse_dates=["release_date"])

    df["release_year"] = df["release_date"].dt.year.fillna(0).astype(int)
    df.drop(["release_date"], axis=1, inplace=True, errors="ignore")

    # filter dataset
    logger.info('Shape of IMDB movies before filtering {}'.format(df.shape))
    df = df[(df.release_year >= 2000) &
            (df.original_language == 'en') &
            (df.imdb_id.notnull()) &
            (df.poster_path.notnull()) &
            (df.status == 'Released') &
            (df.imdb_votes >= 20) &
            (df.vote_count > 0) &
            (df.spoken_languages.str.contains('English'))
            ]

    # drop movies that are too short to be a movie
    # do this as a separate step otherwise filter is too slow
    df = df[ df['runtime'] >= 60.0]
    logger.info('Shape of IMDB movies after filtering {}'.format(df.shape))

    # drop rows with more than 25% missing values
    df = df.loc[df.isna().mean(axis=1) < 0.25, :]

    # drop duplicates id's
    df = df.drop_duplicates(['imdb_id'])
    logger.info(
        'Shape of IMDB movies after removing duplicates {}'.format(df.shape))

    # add a column to convert imdb_id into an integer so we can join the tables
    df['imdbId'] = df['imdb_id'].str[2:].astype('int32')

    # optimize datatypes to reduce memory footprint
    df = reduce_mem_usage(df)

    return df

def clean_imdb_extra(
        path_raw: str
) -> pd.DataFrame:
    """
    Load and clean raw extra IMDB/TMDB data.

    INPUT:
    path_raw: raw folder where extra IMDB/TMDB data are stored in csv format

    OUTPUT:
    df: cleaned extra IMDB/TMDB dataset
    """

    logger.info('Cleaning extra TMDB/IMDB movie content data...')
    print('Cleaning extra TMDB/IMDB movie content data...')

    # load dataset with only columns we want to merge with the movie dataset
    cols_of_interest = ['imdb_id', 'homepage', 'Star1', 'Star2', 'Star3',
                        'Star4', 'backdrop_path']
    df = pd.read_csv(
        path_raw,
        usecols=cols_of_interest,
    )
    df.columns = df.columns.str.lower()
    logger.info('Shape of extra IMDB movies after import {}'.format(df.shape))

    # drop records where id is null
    df.dropna(subset=['imdb_id'], inplace=True)
    text = 'Shape of extra IMDB movies after removal of null ids {}'
    logger.info(text.format(df.shape))

    # convert imdb_id to integer to merge with the main movie dataset later
    df['imdbId'] = df['imdb_id'].str[2:].astype('int32')
    del df['imdb_id']

    # remove any duplicate id's
    df = df.drop_duplicates(['imdbId'])
    text = 'Shape of extra IMDB movies after removing duplicate ids {}'
    logger.info(text.format(df.shape))

    logger.info('Dataset columns: {}'.format(df.columns.tolist()))

    return df

def clean_movielens_links(
        path_raw: str,
        df_movies: pd.DataFrame
) -> pd.DataFrame:
    """
    Load and clean raw MovieLens links data

    The `links` dataset map id's from IMDB,
    TMDB, and MovieLens datasets allowing datasets to be linked.

    INPUT:
    df_movies: filtered and cleaned movie content dataset

    OUTPUT:
    df: cleaned movielens links dataset
    """

    logger.info('Cleaning MovieLens LINKS data...')
    print('Cleaning MovieLens LINKS data...')

    # load data
    cols_of_interest = ['movielens_id', 'imdbId']
    df = pd.read_csv(
        path_raw,
        usecols=cols_of_interest,
        names=cols_of_interest,
        header=0,
        dtype='int32',
    )

    # drop duplicates id's
    df = df.drop_duplicates(['imdbId'])
    logger.info('Shape of links after removing duplicates {}'.format(df.shape))

    # filter links dataset to movies in scope
    logger.info('Shape of links before filtering {}'.format(df.shape))
    df = df.merge(df_movies, on='imdbId', how='inner')[cols_of_interest]
    logger.info('Shape of links after filtering {}'.format(df.shape))

    return df


def clean_movielens_ratings(
        path_raw: str,
        df_links: pd.DataFrame
) -> pd.DataFrame:
    """
    Load and clean RAW MovieLens ratings data

    The `ratings` datasets contain movie ratings by user id and movie id

    INPUT:
    path_raw: raw folder where data are stored in csv format
    df_links: filtered and cleaned movie links dataset

    OUTPUT:
    df: cleaned movielens ratings dataset
    """

    logger.info('Cleaning MovieLens RATINGS data...')
    print('Cleaning MovieLens RATINGS data...')

    # load data
    dtypes = {'user_id': 'int32', 'movielens_id': 'int32', 'rating': 'float16'}
    column_names = ['user_id', 'movielens_id', 'rating', 'timestamp']
    df = pd.read_csv(
        path_raw,
        usecols=column_names,
        names=column_names,
        header=0,
        dtype=dtypes)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # filter dataset to movies in scope
    logger.info('Shape of ratings before filtering {}'.format(df.shape))
    df = df.merge(df_links, on='movielens_id', how='inner')
    logger.info('Shape of ratings after filtering {}'.format(df.shape))

    # to make sure a user rates a movie only once
    df = df.drop_duplicates(['user_id', 'imdbId'])
    logger.info(
        'Shape of ratings after removing duplicates {}'.format(df.shape))

    # popular movie filter - a movie got rated at least 10 times in MovieLens
    df_movies_cnt = pd.DataFrame(
        df.groupby('movielens_id').size(), columns=['count'])
    popular_movies = list(set(df_movies_cnt.query('count > 10').index))
    movies_filter = df['movielens_id'].isin(popular_movies).values

    # active user filter - user rated at least 20 movies in MovieLens
    df_users_cnt = pd.DataFrame(
        df.groupby('user_id').size(), columns=['count'])
    active_users = list(set(df_users_cnt.query('count > 20').index))
    users_filter = df['user_id'].isin(active_users).values

    # filter the data
    df = df[movies_filter & users_filter]
    logger.info('Shape of ratings after additional filtering: {}'.format(
        df.shape))

    del df_movies_cnt, df_users_cnt

    return df


def clean_movielens_tags(
        path_raw: str,
        df_links: pd.DataFrame,
        df_movies: pd.DataFrame
) -> pd.DataFrame:
    """
    Load and clean RAW MovieLens tags data

    The `tags` datasets contain movie tags added by users,
    e.g. 'fantasy', 'family', 'animation'.
    Create a new dataset consolidating all tags as one long string e.g.
    'fantasy, family, animation'.
    In addition, add text and names of actors, producers,
    etc. from the movie datasets to the tags.
    Duplicate tags per movie are left,
    as it will highlight the importance of that tag related to the movie

    INPUT:
    path_raw: raw folder where data are stored in csv format
    df_links: filtered and cleaned movie links dataset
    df_movies: filtered and cleaned movie dataset

    OUTPUT:
    df: cleaned TAGS dataset
    """

    logger.info('Cleaning MovieLens TAGS data...')
    print('Cleaning MovieLens TAGS data...')

    # load data
    cols = ['user_id', 'movielens_id', 'tag', 'timestamp']
    df = pd.read_csv(path_raw, names=cols, header=0)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['tag'] = df['tag'].str.lower()

    # create a tag summary per movie
    df_tag_summary = (df.groupby('movielens_id')['tag']
                      .apply(list).str.join(' ').to_frame().reset_index())

    # Filter tags to movies in scope, add IMDB movie id,
    # and drop the movielens_id
    df_tag_summary = (
        df_tag_summary.merge(df_links, on='movielens_id', how='inner'))
    del df_tag_summary['movielens_id']

    # add all movie text columns
    text_cols = ['imdbId', 'title', 'overview', 'tagline', 'genres',
                 'director', 'producers', 'star1', 'star2', 'star3', 'star4',
                 'production_companies']
    df_tag_summary = df_movies[text_cols].merge(df_tag_summary,
                                                on='imdbId',
                                                how='left')
    df_tag_summary = df_tag_summary.fillna('')

    # create one string column containing all texts
    text_cols = ['title', 'overview', 'tagline', 'genres', 'director',
                 'producers', 'star1', 'star2', 'star3', 'star4',
                 'production_companies', 'tag']
    df_tag_summary['all_texts'] = (
        df_tag_summary[text_cols].agg(' '.join, axis=1))
    df_tag_summary.drop(text_cols, axis=1, inplace=True)

    return df_tag_summary


def parse_args():
    """ default arguments """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log',
        dest='log',
        nargs='?',
        default=path_log_wrangling,
        help='provide path and name of log file')

    parser.add_argument(
        '--path_imdb_raw',
        nargs='?',
        default=path_imdb_raw,
        help='provide RAW path and name of movie content dataset')

    parser.add_argument(
        '--path_imdb_raw_extra',
        nargs='?',
        default=path_imdb_raw_extra,
        help='provide RAW path and name of extra movie content dataset')

    parser.add_argument(
        '--path_imdb_clean',
        nargs='?',
        default=path_imdb_clean,
        help='provide CLEAN path and name of movie content dataset')

    parser.add_argument(
        '--path_links_raw',
        nargs='?',
        default=path_links_raw,
        help='provide RAW path and name of MovieLens LINKS dataset')

    parser.add_argument(
        '--path_links_clean',
        nargs='?',
        default=path_links_clean,
        help='provide CLEAN path and name of MovieLens LINKS dataset')

    parser.add_argument(
        '--path_ratings_raw',
        nargs='?',
        default=path_ratings_raw,
        help='provide RAW path and name of MovieLens RATINGS dataset')

    parser.add_argument(
        '--path_ratings_clean',
        nargs='?',
        default=path_ratings_clean,
        help='provide CLEAN path and name of MovieLens RATINGS dataset')

    parser.add_argument(
        '--path_tags_raw',
        nargs='?',
        default=path_tags_raw,
        help='provide RAW path and name of MovieLens TAGS dataset')

    parser.add_argument(
        '--path_tags_clean',
        nargs='?',
        default=path_tags_clean,
        help='provide CLEAN path and name of MovieLens TAGS dataset')

    return parser.parse_args()


def enrich_movies(
        df_movies: pd.DataFrame,
        df_ratings: pd.DataFrame,
        df_movies_extra: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich the movie dataset with the latest rating timestamp

    INPUT:
    df_movies (pd.DataFrame): Movie dataset
    df_ratings (pd.DataFrame): Ratings dataset
    df_movies_extra (pd.DataFrame): Movie dataset with extra columns

    OUTPUT:
    df_movies = df_movies
    Enriched with the latest rating timestamp and
    additional columns from the df_movies_extra dataset
    """

    temp = pd.DataFrame(
        df_ratings.groupby('imdbId')['timestamp'].max())
    temp.columns = ['last_rating_timestamp']
    df_movies = df_movies.merge(
        temp,
        on='imdbId',
        how='left')
    del temp

    df_movies = df_movies.merge(
        df_movies_extra,
        on='imdbId',
        how='left')

    # correct full link to images
    df_movies['poster_path'] = 'https://image.tmdb.org/t/p/original' + \
                                df_movies['poster_path']

    df_movies['backdrop_path'] = 'https://image.tmdb.org/t/p/original' + \
                                  df_movies['backdrop_path']

    return df_movies


def save_data(
        df: pd.DataFrame,
        path: str,
        cols_to_keep: List,
        index_col: str = None,
        compression: str = 'snappy',
) -> None:
    """
    Save data in the `data/clean` folder as a parquet file

    Finally, only keep columns needed and set index if required

    INPUT:
    df: dataframe to save
    path: path to save the data
    cols_to_keep: list of columns to keep
    index_col: optional, column to use for index
    """

    # create the `data/clean` directory if it does not yet exist
    os.makedirs(folder_clean, exist_ok=True)

    df = df[cols_to_keep]

    if index_col:
        df.set_index(index_col, inplace=True)
        df.to_parquet(path, index=True, compression=compression)
    else:
        df.to_parquet(path, index=False, compression=compression)

    logger.info('\n Dataset with shape {} saved to location {}'.format(
        df.shape, path))
    logger.info('Dataset columns: {}'.format(df.columns.tolist()))


def main():
    """ main routine """

    logger.info('\n\nTRIGGERING SCRIPT {}...'.format(__file__))

    # get args
    args = parse_args()

    df_movies_clean = clean_imdb(
        path_raw=args.path_imdb_raw,
    )

    df_movies_clean_extra = clean_imdb_extra(
        path_raw=args.path_imdb_raw_extra,
    )

    df_links_clean = clean_movielens_links(
        path_raw=args.path_links_raw,
        df_movies=df_movies_clean,
    )

    df_ratings_clean = clean_movielens_ratings(
        path_raw=args.path_ratings_raw,
        df_links=df_links_clean,
    )

    df_movies_clean = enrich_movies(
        df_movies_clean,
        df_ratings_clean,
        df_movies_clean_extra,
    )

    df_tags_clean = clean_movielens_tags(
        path_raw=args.path_tags_raw,
        df_links=df_links_clean,
        df_movies=df_movies_clean,
    )

    # save data
    cols_to_keep = ['imdbId', 'title', 'overview', 'tagline', 'imdb_rating',
                    'imdb_votes', 'poster_path', 'release_year',
                    'last_rating_timestamp', 'genres',
                    'homepage', 'backdrop_path']

    save_data(
        df=df_movies_clean,
        path=args.path_imdb_clean,
        cols_to_keep=cols_to_keep,
        index_col='imdbId'
    )

    cols_to_keep = ['user_id', 'rating', 'timestamp', 'imdbId']
    save_data(
        df=df_ratings_clean,
        path=args.path_ratings_clean,
        cols_to_keep=cols_to_keep,
        compression='gzip',
    )

    save_data(
        df=df_tags_clean,
        path=args.path_tags_clean,
        cols_to_keep=df_tags_clean.columns,
        index_col='imdbId'
    )

    save_data(
        df=df_links_clean,
        path=args.path_links_clean,
        cols_to_keep=df_links_clean.columns,
    )


if __name__ == '__main__':
    main()
