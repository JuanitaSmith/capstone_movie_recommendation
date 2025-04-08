import logging
import re
import warnings
from pickle import dump, load
from typing import Dict, List

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from pandas import DataFrame
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import (path_log_webapp, path_tfidf, path_tfidf_df,
                 path_user_item_matrix)

# suppress warnings
warnings.filterwarnings('ignore')


class Recommender:
    """


    """
    def __init__(self, activate_logger=True):
        """ Initialize variables and activate logger """

        self.user_item = pd.DataFrame()
        self.num_user_ratings = None
        self.num_movie_ratings = None
        self.tfidf_vectorizer = None
        self.tfidf_df = pd.DataFrame()
        self.df_movies = pd.DataFrame()
        self.df_ratings = pd.DataFrame()
        self.df_tags = pd.DataFrame()

        # activate logging
        print('Activating logger...')
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=path_log_webapp,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            filemode='a',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

    def get_data(
            self,
            filename_movies: str,
            filename_ratings: str,
            filename_tags: str):
        """
        Load cleaned movie and rating parquet files into a pandas dataframe

        Input:
        filename_movies (str): path to the movie parquet file
        filename_ratings (str): path to the `ratings` parquet file
        filename_tags (str): path to the `tags` parquet file

        Output:
        self.df_ratings (DataFrame): pandas dataframe with user ratings
        self.df_movies (DataFrame): pandas dataframe with movie contents
        self.df_tags (DataFrame): pandas dataframe with movie texts
        """

        print('Getting data...')
        self.logger.info(f'Loading {filename_movies}...')
        self.logger.info(f'Loading {filename_ratings}...')
        self.logger.info(f'Loading {filename_tags}...')

        self.df_movies = pd.read_parquet(filename_movies)
        self.df_ratings = pd.read_parquet(filename_ratings)
        self.df_tags = pd.read_parquet(filename_tags)

        # calculate the total number of ratings per users and movies
        self.count_user_ratings()
        self.count_movie_ratings()

        self.logger.info('Movie contents loaded with shape {}'.format(
            self.df_movies.shape))
        self.logger.info('User ratings loaded with shape {}'.format(
            self.df_ratings.shape))
        self.logger.info('Movie Tags loaded with shape {}'.format(
            self.df_tags.shape))

    def get_movie_contents(self, movies: np.array) -> Dict:
        """
        Get movie content for movies we want to recommend

        Extract the necessary columns we need to pass to the webapp

        INPUT:
        movies
        (np.array) -> IMBD ids of movies to recommend in order of importance

        OUTPUT:
        top_movies (Dict) -> Dict with movie ids as keys
        and movie content as values
        """

        cols_to_extract = ['title', 'genres', 'imdb_rating',
                           'imdb_votes', 'tagline', 'overview',
                           'poster_path', 'homepage', 'backdrop_path',
                           'genres']

        # movies are no longer sorted in order of importance
        top_movies = self.df_movies.loc[movies][cols_to_extract]


        # sort again in same sequence as input list
        top_movies = top_movies.loc[movies]

        # top_movies['poster_path'] = 'https://image.tmdb.org/t/p/original' + \
        #                             top_movies['poster_path']
        #
        # top_movies['backdrop_path'] = 'https://image.tmdb.org/t/p/original' + \
        #                             top_movies['backdrop_path']

        top_movies['imdb_rating'] = round(
            top_movies['imdb_rating'], 1).astype(int)

        top_movies['imdb_votes'] =  top_movies['imdb_votes'].astype(int)

        # If tagline is not available,
        # fill with the first 100 characters of overview
        top_movies['tagline'] = np.where(
            top_movies['tagline'].isna(),
            top_movies['overview'].map(lambda x: x[:100]),
            top_movies['tagline'])

        # make `imdbId` a column, and index sequence of importance
        top_movies = top_movies.reset_index()

        top_movies = top_movies.to_dict(orient='index')

        return top_movies

    def get_top_movies(self, n: int) -> Dict:
        """
        Use ranked-based recommendation to find the most popular movies

        Most popular movies are defined by the highest IMDB vote count.
        With ties, sort by the most recent rating timestamp

        INPUT:
        n - (int) the number of top articles to return

        OUTPUT:
        top_articles - (dict) Dict containing top n articles
            * key is integer in sequential order of importance
            * values contain columns needed for webapp cards
        """

        top_movies = self.df_movies.sort_values(
            by=['imdb_votes', 'last_rating_timestamp'],
            ascending=False)[:n]

        # get movie details and convert to dictionary
        top_movies = top_movies.index.tolist()
        top_movies = self.get_movie_contents(top_movies)

        # Return the top movies
        return top_movies

    def create_user_item_matrix(self):
        """
        Build a user-user matrix we can use for similarity calculations

        Return a matrix with user ids as rows and movie ids as columns
        and ratings as the values.

        To handle sparsity, center each user's ratings around 0,
        by deducting the row average
        and then filling the missing values with 0.
        This means missing values are replaced with neutral scores.

        This is not a perfect solution as we lose interpretability,
        but if we use these values only to compare users, it's ok.
        Don't try to predict user ratings.

        NOTE: Do this only once at the beginning of your webapp.

        INPUT:
        df - pandas dataframe containing user id, movie id, and rating

        OUTPUT:
        self.user_item - user item matrix
            * with users as index
            * movies as columns
            * ratings as values
        """

        self.logger.info('Creating user item matrix')

        # unstack cannot work with float16, lets code around it
        df_ratings_tmp = (
            self.df_ratings[['user_id', 'imdbId', 'rating']].copy())
        df_ratings_tmp['rating'] = df_ratings_tmp['rating'].astype(float)

        # build the user-item matrix
        self.user_item = df_ratings_tmp.groupby(['user_id', 'imdbId'])[
            'rating'].max().unstack()

        # reduce memory usage
        self.user_item = self.user_item.astype('float16')

        # To handle sparsity, center data around 0
        # and fill nan with 0 to give it a neural score
        # Get average row rating to get average ratings of users
        avg_ratings = self.user_item.mean(axis=1)

        # center the ratings
        self.user_item = self.user_item.sub(avg_ratings, axis=0)

        # fill nan values with 0
        self.user_item = self.user_item.fillna(0)

        del df_ratings_tmp

    def load_user_item_matrix(self):
        """
        Load user_item matrix

        Try to load the pre-build user-item matrix, to speed up
        the webapp loading time.

        Create the user-item matrix if it does not exist.
        """

        self.logger.info('Loading user item matrix')

        # first try to load the pre-build saved user-item matrix
        try:
            self.user_item = pd.read_parquet(path_user_item_matrix)
        except FileNotFoundError:
            self.create_user_item_matrix()


    def count_user_ratings(self):
        """
        Calculate the total number of times each user rated movies

        INPUT:
        df - pandas dataframe containing all the movie ratings

        OUTPUT:
        self.num_user_ratings - pandas series containing total votes per user
             * index = user_id
             * value: total times movies were rated
        """

        self.num_user_ratings = self.df_ratings.groupby('user_id')[
            'rating'].size().sort_values(ascending=False)

    def count_movie_ratings(self):
        """
        Calculate the total number of times each movie was rated

        Input:
        df - pandas dataframe containing all the movie ratings

        Output:
        self.num_movie_ratings - pandas series containing total votes per user
             * index = movie id
             * value: total times movies were rated
        """

        self.num_movie_ratings = self.df_ratings.groupby('imdbId')[
            'rating'].size().sort_values(ascending=False)

    def find_similar_users(
            self,
            user_id: int,
            top_n: int = 200) -> pd.DataFrame:
        """
        Find the nearest neighbors for an input user using cosine similarity

        Sort neighbors by cosine similarity and their number of total votes

        INPUT:
        user_id - (int) a user_id
        user_item - (pandas dataframe) matrix of users by articles:
                    1's when a user has interacted with an article, 0 otherwise

        OUTPUT:
        knn - (pandas Dataframe) with columns:
            * index (int): incremental counter showing order of neighbor
            importance
            * user_id (int): unique user id assigned to each neighbor
            * similarity (float): cosine similarity score rounded to 4 decimals
            * num_user_ratings (int): total number of votes per user
            (for all movies)
        """

        # find closest neighbors for a user
        input_user_series = self.user_item.loc[user_id].values.reshape(1, -1)

        user_similarities = cosine_similarity(
            input_user_series,
            self.user_item,
        )

        user_similarities_df = pd.DataFrame(user_similarities,
                                            columns=self.user_item.index,
                                            )

        # get top similar users
        knn = user_similarities_df.loc[0].to_frame(name='similarity')
        knn['similarity'] = round(knn['similarity'], 4)
        knn = knn.sort_values(by='similarity', ascending=False)[1:top_n]
        knn = knn.reset_index(names='user_id')

        # get the total number of votes for each neighbor and sort
        knn['num_votes'] = knn.user_id.apply(
            lambda x: self.num_user_ratings.loc[x]).squeeze()
        knn = knn.sort_values(by=['similarity', 'num_votes'],
                              ascending=[False, False])
        return knn

    def get_movies_watched(self, user_id: str) -> np.array:
        """
        Get movies a user has seen already (based on ratings)

        Movies will be sorted by the highest rating of the user,
         and then the latest timestamp to solve ties.

        INPUT:
        user_id: int -> Id of a user

        OUTPUT:
        movies_watched (np.array) -> sorted list of movie_id's
        a user has seen in order of importance
        """

        movies_watched = self.df_ratings[
            self.df_ratings['user_id'] == user_id].copy()
        movies_watched['rating'] = movies_watched['rating'].astype(float)
        movies_watched = movies_watched.sort_values(by=['rating', 'timestamp'],
                                                    ascending=False)
        movies_watched = np.array(movies_watched['imdbId'])

        return movies_watched

    def user_user_recommendations(
            self,
            user_id: int,
            top_n: int = 10, ) -> tuple[Dict, str]:
        """
        Get top movies to recommend to user

        Steps:
        1) Get movies the input user has watched
        2) Get the closest neighbors user_id
        3) Start with the first neighbor:
           - 3.1) Get the list of movies the neighbor watched in order of importance
           - (rating given by the neighbor and the latest timestamp)
           - 3.2) Remove movies already watched by input user
           - 3.3) Add movies in the order of importance to a recommendation list
           - 3.4) Continue down the list until top_n movies are found
        4) Get movie columns we need to pass to the webapp
        5) Convert to dictionary

        Following the steps above
        means we could end up recommending the movies from the top neighbor only!
        (Which is ok)

        INPUT:
        user_id - (int) a user id
        top_n - (int) the number of movies to recommend

        OUTPUT:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title
        """

        search_text = ''

        # list of movies to recommend
        recs = []

        # Get movies input user has watched (in order of importance)
        input_movies_watched = self.get_movies_watched(user_id)

        # get nearest neighbors
        neighbors_df = self.find_similar_users(user_id)

        for neighbor in neighbors_df['user_id'].tolist():

            self.logger.info('Nearest neighbor {} selected'.format(neighbor))

            # get movies the neighbor watched (in order of importance)
            neighbor_movies_watched = self.get_movies_watched(neighbor)

            # remove movies input user watched while preserving the order
            mask = ~np.isin(neighbor_movies_watched, input_movies_watched)
            new_recs = neighbor_movies_watched[mask]

            # add movies to our recommendation list
            recs.extend(new_recs)

            # stop if recommendations exceed
            # the number of required recommendations
            if len(recs) >= top_n:
                break

        # Select only the top_n
        recs = list(recs[:top_n])

        top_movies = self.get_movie_contents(recs)

        return top_movies, search_text

    def tokenize(self, text: str) -> List:
        """ Summarize text into words

        Most important functions:
        - Summarize url links starting with http or www to a common phrase 'url
        - Summarize email addresses to a common phrase 'email'
        - Get rid of new lines `\n'
        - Remove all words that are just numbers
        - Remove all words that contain numbers
        - Cleanup basic punctuation like '..', '. .'
        - Remove punctuation
        - Remove words that are just 1 character long after removing punctuation
        - Use lemmatization to bring words to the base

        INPUT:
            text: string, Text sentences to be split into words

        OUTPUT:
            clean_tokens: list, List containing most crucial words
        """

        # Replace urls starting with 'https' with placeholder
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        # replace urls with a common keyword
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, 'url')

        # Replace urls starting with 'www' with placeholder
        url_regex = 'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, 'url')

        # replace emails with placeholder
        email_regex = '([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})'
        detected_emails = re.findall(email_regex, text)
        for email in detected_emails:
            text = text.replace(email, 'email')

        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("..", ".")
        text = text.replace(". .", ".")
        text = text.replace(" ,.", ".")

        text = re.sub(r'\s+', ' ', text).strip()

        # normalize text by removing punctuation, remove case and strip spaces
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        text = text.lower().strip()

        # remove numbers
        text = re.sub(r'\d+', '', text)

        #  split sentence into words
        tokens = word_tokenize(text)

        # Remove stopwords, e.g. 'the', 'a',
        tokens = [w for w in tokens if w not in stopwords.words("english")]

        # take words to their core, e.g., children to child
        lemmatizer = WordNetLemmatizer()
        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok, wordnet.VERB)
            # ignore tokens that have only 1 character or contains numbers
            if len(clean_tok) >= 2 & clean_tok.isalpha():
                clean_tokens.append(clean_tok)

        return clean_tokens

    def create_word_count_matrix(
            self,
            column: str = 'all_texts',
            max_features: int = 5000,
            min_df: int = 3,
            max_df: int = 0.7) -> tuple[pd.DataFrame, TfidfVectorizer()]:
        """
        Create a word count matrix for a dataframe column containing text

        INPUT:
        column: string -> column to convert to word counts
        max_features: int -> maximum number of features to use
        min_df: int -> minimum number of occurrences of a word
        max_df: int -> maximum number of occurrences of a word

        OUTPUT:
        tfidf_df: pandas dataframe containing the word count matrix
        tfidf_vectorizer: tfidf trained model object

        """

        # Create a word count matrix by article
        tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            tokenizer=self.tokenize,
            token_pattern=None,
            max_features=max_features)

        vectorized_data = tfidf_vectorizer.fit_transform(self.df_tags[column])
        tfidf_df = pd.DataFrame(
            vectorized_data.toarray(),
            columns=tfidf_vectorizer.get_feature_names_out(),
            index=self.df_tags.index)

        return tfidf_df, tfidf_vectorizer

    def get_user_interests(self, user_id: str, top_n: int = 10) -> List:
        """
        Get a list of keywords to describe what a user is most interested in

        INPUT:
        user_id: int -> Id of a user
        top-n: int -> Number of most used keywords to return

        OUTPUT:
        top_keywords: list, List containing most used nouns and verbs
        """

        # get all the documents titles a user has read
        docs_read = (self.df_ratings[
                         self.df_ratings['user_id'] == user_id][
                         'title']
                     .tolist())

        # keep only keywords that are nouns and verbs
        keywords = []
        for title in docs_read:
            keyword = self.tokenize(title)
            pos_tags = nltk.pos_tag(keyword)
            for tag in pos_tags:
                if tag[1] in ['NN', 'NNP', 'NNS', 'VB']:
                    keywords.append(tag[0])

        # get the top n keywords
        top_keywords = pd.value_counts(keywords)[:top_n].index.tolist()

        return top_keywords

    def make_content_recommendations(
            self,
            tfidf_df: pd.DataFrame,
            tfidf_vectorizer: TfidfVectorizer(),
            input_search_text: str,
            user_id: int,
            top_n=10) -> Dict:
        """
        Content-based recommendations based on text-based similarity

        User input any search text

        INPUT:
        input_search_text: string, any text a user input to search for documents
        user_id: integer, id of the user we make recommendations for
        tfidf: pandas dataframe containing the word count matrix
        tfidf_vectorizer: word count model,
        to vectorize user input to the same standard

        OUTPUT:
            top_articles - (dict) Dict containing top n articles
                * key is integer in sequential order of importance
                * values contain columns needed for webapp cards
        """

        top_movies = {}

        # convert input text to the fitted tfidf model
        input_search_text = (
            tfidf_vectorizer.transform([input_search_text]))

        # find cosine similarity between the input search text to each document
        similarity = cosine_similarity(input_search_text, tfidf_df)
        cosine_similarity_df = pd.DataFrame(
            similarity,
            columns=tfidf_df.index)

        # get the most similar records, ranked by the highest similarity
        cosine_similarity_df = cosine_similarity_df.loc[0].sort_values(
            ascending=False).reset_index()

        # to handle ties, sort by similarity and number of IMDB votes
        df_temp_movies = self.df_movies[['imdb_votes']]
        cosine_similarity_df = cosine_similarity_df.merge(df_temp_movies,
                                                          on='imdbId',
                                                          how='left').fillna(0)
        cosine_similarity_df.columns = ['imdbId', 'similarity', 'imdb_votes']
        cosine_similarity_df = cosine_similarity_df.sort_values(
            by=['similarity', 'imdb_votes'], ascending=False)
        top_movies = np.array(cosine_similarity_df['imdbId'])

        # get documents input user has seen already
        input_movies_watched = self.get_movies_watched(user_id)

        # remove documents the user has seen but preserving ranking
        mask = ~np.isin(top_movies, input_movies_watched)
        top_movies = top_movies[mask]

        # get movie content like title, overview etc
        top_movies = self.get_movie_contents(top_movies[:top_n])

        return top_movies

    def matrix_factorization(self, user_id, top_n=10):
        """
        Use SVD matrix factorization to get recommendations

        INPUT: user_id: integer, id of the user we make recommendations for
        top_n: int -> Number of documents to recommend

        OUTPUT:
        prediction_df -> DataFrame containing recommendations

        """
        # create a user-item matrix
        user_item_matrix = self.load_user_item_matrix()

        # Decompose the matrix using training dataset with SVD with k latent features
        U, sigma, Vt = svds(np.array(user_item_matrix), k=400)

        # correct shape of s (latent features)
        sigma = np.diag(sigma)

        # predict which documents users will enjoy using the dot product
        prediction = np.abs(np.around(np.dot(np.dot(U, sigma), Vt), 0))
        prediction_df = pd.DataFrame(prediction,
                                     columns=user_item_matrix.columns,
                                     index=user_item_matrix.index)

        # filter predictions to input user
        prediction_df = prediction_df.loc[user_id].sort_values(ascending=False)

        # keep records where the predicted user will read the document
        prediction_df = prediction_df[prediction_df > 0]

        # Get and remove documents the user read already
        movies_watched = self.get_movies_watched(user_id)
        prediction_df.drop(movies_watched, inplace=True, errors='ignore')

        if prediction_df.shape[0] > 0:

            # merge and sort predictions and interactions
            top_movie_interactions = (
                self.df_ratings.movie_id.value_counts())

            similar_movies = pd.concat(
                [prediction_df, top_movie_interactions],
                axis=1,
                join='inner')

            similar_movies.columns = ['similarity', 'num_interactions']
            similar_movies.sort_values(by=['similarity', 'num_interactions'],
                                         ascending=False, inplace=True)

            # add contents to our recommendations
            top_movies = self.get_movie_contents(similar_movies[:top_n])

        else:
            print("No recommendations found")
            self.logger.info("No recommendations found")

        return prediction_df

    def generate_tfidf_vectorizer(
            self,
            path_tfidf: str,
            path_tfidf_df: str,
            text_column: str ='all_texts'):
        """ Generate and save tfidf vector object and matrix"""

        # create word count matrix
        self.logger.info('Generating tfidf word count matrix and model...')
        print('Generating tfidf word count matrix and model...')
        self.tfidf_df, self.tfidf_vectorizer = (
            self.create_word_count_matrix(column=text_column))

        # save to file
        if self.tfidf_df.shape[0] > 0:
            print('Saving model to path {}'.format(path_tfidf))
            self.logger.info('Saving model to path {}'.format(path_tfidf))
            dump(self.tfidf_vectorizer, open(path_tfidf, 'wb'))

            print('Saving matrix to path {}'.format(
                path_tfidf_df))
            self.logger.info('Saving matrix to path {}'.format(
                path_tfidf_df))
            self.logger.info('TFIDF_DF shape: {}'.format(
                self.tfidf_df.shape))
            self.tfidf_df.to_parquet(path_tfidf_df)

    def load_tfidf_vectorizer(self):
        """ Load tfidf vectorizer and matrix

        As the tfidf vector takes a long time
        to run due to the custom tokenizer,
        the tokenizer object and matrix are loaded from disk if it exists.
        If it does not exist,
        the recommender will create it the first time it runs.

        Alternatively,
        A separate script `src/preprocessing_nlp.py` can be scheduled
        to run periodically to create the file upfront
        and to incorporate new users and movies.
        """

        try:
            self.logger.info('Loading tfidf vectorizer...')
            self.tfidf_df = pd.read_parquet(path_tfidf_df)
            self.tfidf_vectorizer = load(open(path_tfidf, 'rb'))
        except FileNotFoundError:
            self.generate_tfidf_vectorizer(
                path_tfidf,
                path_tfidf_df,
                text_column='all_texts')
