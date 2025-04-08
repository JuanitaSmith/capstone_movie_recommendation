import os

# DIRECTORY STRUCTURE - FOLDER NAMES
folder_data = 'data'
folder_raw = 'data/raw'
folder_clean = 'data/clean'
folder_embeddings = 'data/embeddings'
folder_logs = 'logs'
folder_scripts = 'src'
folder_models = 'models'
folder_imdb_tmdb = 'imdb_tmdb'
folder_imdb_tmdb_extra = 'imdb_tmdb_extra'
folder_movielens = 'movielens'
folder_movielens_extract = 'movielens/ml-32m'

# LOG FILE NAMES
filename_log_webapp = 'webapp.log'
filename_log_process_data = 'wrangling.log'

# FILE/TABLE NAMES
filename_imdb_tmdb_raw = 'imdb_tmdb.csv'
# filename_imdb_tmdb_raw_extra = 'imdb_tmdb_extra.csv'
filename_imdb_tmdb_clean = 'imdb_tmdb.parquet'
filename_movielens_ratings_raw = 'ratings.csv'
filename_movielens_ratings_clean = 'ratings.parquet.gzip'
filename_movielens_links_raw = 'links.csv'
filename_movielens_links_clean = 'links.parquet'
filename_movielens_movies = 'movies.csv'
filename_movielens_tags_raw = 'tags.csv'
filename_movielens_tags_clean = 'tags.parquet'
filename_tfidf = 'tfidf.pkl'
filename_tfidf_df = 'tfidf_df.parquet'
filename_user_item_matrix = 'user_item_matrix.parquet'

# URL where to fetch data from
url_kaggle_tmdb = 'alanvourch/tmdb-movies-daily-updates'
url_kaggle_tmdb_extra = \
    'shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m'
url_movielens = 'https://files.grouplens.org/datasets/movielens/ml-32m.zip'

# FILE PATHS

# logs
path_log_wrangling = os.path.join(
    folder_logs,
    filename_log_process_data)

path_log_webapp = os.path.join(
    folder_logs,
    filename_log_webapp)

# MovieLens links
path_links_raw = os.path.join(
    folder_raw,
    folder_movielens_extract,
    filename_movielens_links_raw)

path_links_clean = os.path.join(
    folder_clean,
    filename_movielens_links_clean)

# MovieLens tags
path_tags_raw = os.path.join(
    folder_raw,
    folder_movielens_extract,
    filename_movielens_tags_raw)

path_tags_clean = os.path.join(
    folder_clean,
    filename_movielens_tags_clean)

# MovieLens ratings
path_ratings_raw = os.path.join(
    folder_raw,
    folder_movielens_extract,
    filename_movielens_ratings_raw)

path_ratings_clean = os.path.join(
    folder_clean,
    filename_movielens_ratings_clean)

# MovieLens contents
path_movielens_movies = os.path.join(
    folder_raw,
    folder_movielens_extract,
    filename_movielens_movies)

# IMDB contents
path_imdb_raw = os.path.join(folder_raw,
                             folder_imdb_tmdb,
                             filename_imdb_tmdb_raw)

path_imdb_clean = os.path.join(folder_clean,
                               filename_imdb_tmdb_clean)

path_imdb_raw_extra = os.path.join(
    folder_raw,
    folder_imdb_tmdb_extra,
    filename_imdb_tmdb_raw)

# Word count vectorizer and matrix
path_tfidf = os.path.join(folder_models, filename_tfidf)
path_tfidf_df = os.path.join(folder_models, filename_tfidf_df)
path_user_item_matrix = os.path.join(folder_models, filename_user_item_matrix)

# Create the main project directory structure
# os.makedirs(folder_raw, exist_ok=True)
# os.makedirs(folder_clean, exist_ok=True)
# os.makedirs(folder_logs, exist_ok=True)
# os.makedirs(folder_models, exist_ok=True)
