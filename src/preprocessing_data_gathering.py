"""
This script downloads and unzips movie data automatically to support
my movie recommendation webapp.

IMPORTANT run from the main project root using the command
`python -m src.preprocessing_data_gathering`
"""

import argparse
import logging
import os
import sys
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import kaggle

from src import (filename_imdb_tmdb_raw, folder_imdb_tmdb, folder_movielens,
                 folder_raw, path_log_wrangling, url_kaggle_tmdb,
                 url_movielens, url_kaggle_tmdb_extra, folder_imdb_tmdb_extra)

# activate logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=path_log_wrangling,
    format='%(asctime)s %(levelname)-8s %(message)s',
    filemode='w',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def check_tmdb_folder(raw_folder, tmdb_folder):
    """ Check if the `data/raw/imdb_tmdb` folder exists. If not, create it. """

    path = os.path.join(raw_folder, tmdb_folder)
    os.makedirs(path, exist_ok=True)

    return path


def gather_imdb_tmdb(
        path_raw: str,
        path_imdb_tmdb: str,
        url: str,
        rename_trigger: str
) -> None:

    """
    Download IMDB/TMDB movie content data from Kaggle if it does not yet exist

    Data source selected for this project can be found
    [here](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates)

    This dataset contains extensive metadata to support
    content-based recommendations.

    To use the Kaggle API, you need to install the kaggle package by running:
    `pip install kaggle`

    Before accessing the API, you need to authenticate using an API token.
    Follow the steps below:

    1) Go to the 'Account' tab on your Kaggle profile.
    2) Click 'Create New Token'.
    This will download a file named `kaggle.json`
    containing your API credentials.
    3) Move the `kaggle.json` file to the appropriate location:
        * Linux/OSX: `~/.kaggle/kaggle.json`
        * Windows: `c:/Users/<username>/.kaggle/kaggle.json`

    IMPORTANT:
    Ensure `kaggle.json` is in the location `~/.kaggle/kaggle.json` to use the API.

    Alternatively, the data can be downloaded manually using the above link
    and saved in the folder `data/raw/imdb_tmdb/imdb_tmdb.csv`

    To refresh the data, manually delete the folder `data/raw/imdb_tmdb`.

    INPUT:
    path_raw: path to the raw data folder
    path_imdb_tmdb: folder under the raw folder to
     where IMDB/TMDB data will be downloaded
    url: URL from where to download kaggle TMDB data
    """

    logger.info('Extracting TMDB/IMDB movie content...')
    print('Extracting TMDB/IMDB movie content...')

    path = check_tmdb_folder(path_raw, path_imdb_tmdb)

    # Check if the destination folder contains any files.
    # If so, skip extraction
    if (os.path.isdir(path) and
            len(os.listdir(path)) == 1):
        logger.info('File {} already exist'.format(path))
        print('File {} already exist'.format(path))
    else:
        # authenticate kaggle API
        try:
            kaggle.api.authenticate()
        except kaggle.api.auth.AuthenticationError:
            text = 'API token file `kaggle.json` should exist in `~/.kaggle`'
            logger.error('Kaggle authentication failed. {}'.format(text))
            sys.exit()

        # download file
        try:
            # Download TMDB/IMDB movie metadata file from Kaggle
            logger.info('Downloading TMDB/IMDB movie contents at {}'.format(
                url))
            print('Downloading TMDB/IMDB movie contents at {}'.format(
                url))
            kaggle.api.dataset_download_files(
                url,
                path=path,
                unzip=True)
            logger.info(
                'TMDB/IMDB download successful to path {}'.format(path))
            print(
                'TMDB/IMDB download successful to path {}'.format(path))
        except Exception as e:
            logger.error('Kaggle TMDB file download failed. {}'.format(e))
            print('Kaggle TMDB file download failed. {}'.format(e))
            sys.exit(1)

        # rename file name
        try:
            for filename in os.listdir(path):
                file_old = os.path.join(path, filename)
                file_new = os.path.join(path, filename_imdb_tmdb_raw)
                if filename.startswith(rename_trigger):
                    os.rename(file_old, file_new)
                    logger.info('File renamed from {} to {}'.format(
                        file_old, file_new))
                    print('File renamed from {} to {}'.format(
                        file_old, file_new))
        except Exception as e:
            logger.error(
                'File renamed from {} to {} failed with message {}'.format(
                    file_old, file_new, e))
            print(
                'File renamed from {} to {} failed with message {}'.format(
                    file_old, file_new, e))
            sys.exit(1)

        assert len(os.listdir(path)) == 1


def gather_movielens(
        url: str,
        path_raw: str,
        path_movielens: str) -> None:
    """
    Extract MovieLens data to location `data/raw/movielens`.

    Dataset `ml-32m.zip` can be downloaded manually from Grouplens
    [here](http://files.grouplens.org/datasets/movielens/)

    MovieLens data `ratings.csv` contains ratings of movies by user id and
    movie id to support collaborative filtering recommendations.

    The zip file also contains a conversion file `links.csv`
    which links id's of MovieLens,
    IMDB, and TMDB together, giving us a lot of freedom
    to blend data from multiple sources.

    INPUT:
    path_raw: path to the raw data folder
    path_movielens: folder under the raw folder to
     where MovieLens data will be downloaded
    url: URL from where to download MovieLens data
    """

    logger.info('Extracting MovieLens data...')
    print('Extracting MovieLens data...')

    # Download zip file and extract files
    folder_movielens_raw = os.path.join(path_raw, path_movielens)
    if os.path.isdir(folder_movielens_raw) and len(
            os.listdir(folder_movielens_raw)) >= 1:
        logger.info('File {} already exist'.format(folder_movielens_raw))
        print('File {} already exist'.format(folder_movielens_raw))
    else:
        try:
            with urlopen(url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall(folder_movielens_raw)
            logger.info('MovieLens download successful to path {}'.format(
                folder_movielens_raw))
            print('MovieLens download successful to path {}'.format(
                folder_movielens_raw))
        except Exception as e:
            logger.error(
                'MovieLens extraction failed with message {}'.format(e))
            print('MovieLens extraction failed with message {}'.format(e))
            sys.exit(1)

    assert len(os.listdir(folder_movielens_raw)) >= 1


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
        default=folder_imdb_tmdb,
        help='provide path and name of movie content file')

    parser.add_argument(
        '--path_imdb_raw_extra',
        nargs='?',
        default=folder_imdb_tmdb_extra,
        help='provide path and name of movie content file')

    parser.add_argument(
        '--path_movielens',
        nargs='?',
        default=folder_movielens,
        help='provide path and name of MovieLens data folder')

    parser.add_argument(
        '--path_raw',
        nargs='?',
        default=folder_raw,
        help='provide path and name of movie content file')

    parser.add_argument(
        '--url_tmdb',
        nargs='?',
        default=url_kaggle_tmdb,
        help='provide URL to download TMDB data from Kaggle')

    parser.add_argument(
        '--url_tmdb_extra',
        nargs='?',
        default=url_kaggle_tmdb_extra,
        help='provide URL to download extra TMDB data from Kaggle')

    parser.add_argument(
        '--url_movielens',
        nargs='?',
        default=url_movielens,
        help='provide URL to download MovieLens data')

    return parser.parse_args()


def main():
    """ main routine """

    logger.info('\n\nTRIGGERING SCRIPT {}...'.format(__file__))

    # get args for default file paths
    args = parse_args()

    # get movie content data
    gather_imdb_tmdb(
        url=args.url_tmdb,
        path_raw=args.path_raw,
        path_imdb_tmdb=args.path_imdb_raw,
        rename_trigger="TMDB",
    )

    # get additional movie content data
    gather_imdb_tmdb(
        url=args.url_tmdb_extra,
        path_raw=args.path_raw,
        path_imdb_tmdb=args.path_imdb_raw_extra,
        rename_trigger="IMDB TMDB"
    )

    # get movie ratings
    gather_movielens(
        url=args.url_movielens,
        path_raw=args.path_raw,
        path_movielens=args.path_movielens
    )


if __name__ == '__main__':
    main()
