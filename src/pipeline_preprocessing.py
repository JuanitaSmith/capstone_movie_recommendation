"""
Script creates a pipeline to run all data gathering,
cleaning and preprocessing tasks
needed to prepare data for the webapp movie recommender

Finally, unit tests are run

IMPORTANT run from the main project root using the command
`python -m src.pipeline_preprocessing`
"""

from src import (preprocessing_data_gathering,
                 preprocessing_data_cleaning,
                 preprocessing_nlp)

import unittest
from tests.test_recommender import TestRecommender

if __name__ == '__main__':

    # gather and clean data
    preprocessing_data_gathering.main()
    preprocessing_data_cleaning.main()
    preprocessing_nlp.main()

    # unit test
    unittest.main(failfast=True)
