"""
Unit tests for data gathering, cleaning and preprocessing tasks

To run this test:
use the command `python -m unittest tests.test_recommender` in the terminal
"""

import unittest
from src import (path_imdb_clean,
                 path_ratings_clean,
                 Recommender,
                 path_tags_clean)


class TestRecommender(unittest.TestCase):
    def setUp(self):
        """ code that runs before every test """

        # instantiating this class will load and clean the data
        self.r = Recommender(activate_logger=False)
        self.r.get_data(path_imdb_clean, path_ratings_clean, path_tags_clean)

    def test_load_data(self):

        self.assertEqual(self.r.df_ratings.shape[1],
                         4,
                         'Interaction dataset should have 4 columns')

        self.assertEqual(self.r.df_movies.shape[1],
                         17,
                         'Content dataset should have 16 columns')

        self.assertEqual(self.r.df_tags.shape[1],
                         1,
                         'Tags dataset should have 1 column')

    def test_clean_data(self):

        # make sure contents contain no duplicate indexes
        self.assertEqual(
            first=self.r.df_movies.index.duplicated().any(),
            second=False,
            msg='Contents have duplicate indexes'
        )

        self.assertIn(
            'user_id',
            self.r.df_ratings.columns.tolist(),
            'Ratings dataset should have column "user_id"')


    def test_user_item_matrix(self):
        """ Testing creation of the user-item matrix """

        text = 'User-Item matrix should have rows {} = number of users {}'
        self.r.load_user_item_matrix()
        unique_users = self.r.df_ratings['user_id'].nunique()
        self.assertEqual(
            self.r.user_item.shape[0],
            unique_users,
            text.format(
                self.r.user_item.shape, unique_users))

    def test_count_user_ratings(self):
        """ Testing the creation of total ratings by user """

        text = '`count_user_ratings` should have rows {} = number of users {}'
        unique_users = self.r.df_ratings['user_id'].nunique()
        self.assertEqual(
            self.r.num_user_ratings.shape[0],
            unique_users,
            text.format(
                self.r.num_user_ratings.shape[0], unique_users))

    def test_user_user_recommendations(self):
        """ Nearest neighbor test for users 103013 and 150189 """

        self.r.load_user_item_matrix()
        df = self.r.find_similar_users(103013, top_n=2)
        nearest_neighbor = df.loc[0, 'user_id']
        self.assertEqual(
            nearest_neighbor,
            second=2704,
            msg="Nearest neighbor for user 103013 should be 2704")

        self.r.load_user_item_matrix()
        df = self.r.find_similar_users(150189, top_n=2)
        nearest_neighbor = df.loc[0, 'user_id']
        self.assertEqual(
            nearest_neighbor,
            second=109980,
            msg="Nearest neighbor for user 150189 should be 109980")

        top_movies = self.r.user_user_recommendations(103013, 1)
        top_movie = top_movies[0][0]['imdbId']
        self.assertEqual(
            top_movie,
            second=318411,
            msg="Top movie id (imdbId) for user 103013 should be 318411")

        top_movies = self.r.user_user_recommendations(150189, 1)
        top_movie = top_movies[0][0]['imdbId']
        self.assertEqual(
            top_movie,
            second=4633694,
            msg="Top movie id (imdbId) for user 103013 should be 318411")

    def test_content_recommendations(self):
        """ content recommendations test using a search term """

        self.r.load_user_item_matrix()
        tfidf_df, tfidf_vectorizer = self.r.create_word_count_matrix(
            column = 'all_texts')

        # make recommendations with the key word 'batman'
        input_text = 'batman'
        top_movies = self.r.make_content_recommendations(
            tfidf_df=tfidf_df,
            tfidf_vectorizer=tfidf_vectorizer,
            input_search_text='batman',
            user_id=150189,
            top_n=1
        )

        top_movie_id = top_movies[0]['imdbId']
        self.assertEqual(top_movie_id,
                         4116284,
                         "Movie with highest cosine similarity "
                         "to 'batman' should be 4116284")
