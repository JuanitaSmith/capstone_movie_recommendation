"""
Script will create tfidf object and word count data matrix
as it takes too long for to do during the webapp experience.

IMPORTANT run from the main project root using the command
`python -m src.preprocessing_nlp`
"""

import argparse
import logging
import os

import nltk

# download all the necessary NLTK packages needed for the recommendation app
nltk.download(['punkt', 'wordnet', 'stopwords',
               'averaged_perceptron_tagger_eng',
               'averaged_perceptron_tagger', 'punkt_tab'])

from src import (Recommender, folder_models, path_imdb_clean,
                 path_log_wrangling, path_ratings_clean, path_tags_clean,
                 path_tfidf, path_tfidf_df, path_user_item_matrix)

# activate logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=path_log_wrangling,
    format='%(asctime)s %(levelname)-8s %(message)s',
    filemode='a',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


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
        '--path_movies',
        nargs='?',
        default=path_imdb_clean,
        help='provide CLEAN path and name of movie IMDB content dataset')

    parser.add_argument(
        '--path_tags',
        nargs='?',
        default=path_tags_clean,
        help='provide CLEAN path and name of MovieLens TAGS dataset')

    parser.add_argument(
        '--path_ratings',
        nargs='?',
        default=path_ratings_clean,
        help='provide CLEAN path and name of MovieLens RATINGS dataset')

    parser.add_argument(
        '--path_tfidf',
        nargs='?',
        default=path_tfidf,
        help='provide path where tfidf model must be stored')

    parser.add_argument(
        '--path_tfidf_df',
        nargs='?',
        default=path_tfidf_df,
        help='provide path where tfidf word count matrix must be stored')

    return parser.parse_args()


def main():
    """ main routine """

    logger.info('\n\nTRIGGERING SCRIPT {}...'.format(__file__))

    # get args
    args = parse_args()

    # instantiate recommender and load cleaned data
    r = Recommender()
    r.get_data(args.path_movies, args.path_ratings, args.path_tags)

    # create the directory if it does not yet exist
    os.makedirs(folder_models, exist_ok=True)

    # create the word count matrix and model and save it
    r.generate_tfidf_vectorizer(
        path_tfidf=args.path_tfidf,
        path_tfidf_df=args.path_tfidf_df,
        text_column='all_texts',
    )

    # create user-item matrix in advance to speed up web app loading
    r.create_user_item_matrix()
    r.user_item.to_parquet(path_user_item_matrix)
    logger.info('User-Item matrix with shape {} are saved at location {}'
                .format(r.user_item.shape, path_user_item_matrix))


if __name__ == '__main__':
    main()
