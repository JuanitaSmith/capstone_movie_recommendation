## Udacity Data Science Nanodegree
---
## Capstone Project: Use LLM to improve recommendation engines


#### Created by: Juanita Smith
#### Last date: March 2025
---

Starbucks wants to uplift their sales by sending out promotions more selectively to only those customers that are more likely to purchase the product.
Who responds better to promotions?
How can we identify the target market better?

### Table of contents

* [1. Project Overview](#1-project-overview)
* [2. Udacity Project requirements](#2-udacity-project-requirements)
* [3. Installation](#3-installation)
* [4. Instructions](#4-instructions)
* [5. Language translator with CHATGPT (optional step)](#5-language-translator-with-chatgpt-optional-step)
* [6. Input File Descriptions](#6-input-file-descriptions)
* [7. Modelling](#7-modelling)
  * [Data cleaning](#data-cleaning)
  * [Modelling approach](#modelling-approach)
    * [Dealing with Imbalance](#dealing-with-imbalance)
    * [Cross-validation](#cross-validation)
    * [Evaluation metrics](#evaluation-metrics)
    * [Model performance](#model-performance)
* [8. Flask Web App](#8-flask-web-app)
* [9. Skills learned:](#9-skills-learned)
* [10. Licensing, Authors, Acknowledgements<a name="licensing"></a>](#10-licensing-authors-acknowledgementsa-namelicensinga)
* [11. References](#11-references)


# 1. Project Definition

## Project Overview

This project was completed as part of the [Data Science Nanodegree](https://www.udacity.com/enrollment/nd025) with Udacity.

As I really enjoyed the recommendations part of this course,
this capstone project will produce a more advanced movie recommender web app,
than what we practiced in class lessons.

To connect with my business partners,
focussing on meaningful communication is as important
to showing technical capability.

Datasets containing over one million movies from TMDB/IMDB were sourced from Kaggle,
and blended with rating data from MovieLens.

As all popular movie platforms like Netflix is user/customer focused,
where an account is needed to watch movies,
and recommendations are personalized to the user,
I will build the web app in the same way to be customer-focused.


Content-based movie recommender to explore the value of LLM's.
Recommend movies with plots similar to those that a user has rated highly.

Information Retrieval:
Return a movie title based on an input plot description

Text Classification:
Predict movie genre based on plot description


Group similar documents

## Problem Statement

During the Udacity data science lessons,
older NLT processing techniques were used from the NLTK library.

Sentences were broken up into words,
and similarity or classification solutions were built by word matching without any schematic meaning or relationships between words or sentences.

It relied on exact keyword-based searches.

Does not understand context,
and does not consider
the same word can have different usages and different contexts.

As the world of AI is currently exploding with the introduction of LLMs,
I want to explore the value of using LLM when doing movie recommendations
and if evaluate if this will improve the quality of movie recommendations.

No Semantic Information: 
Count Vectorizer only captures word counts, not the meaning or context. 
For instance, words like “cat” and “kitten” would have different vectors, even though they are closely related.
Counting word frequencies.
TF-IDF - introduce weights to represent the frequency of words in a document, such as TFIDF

## Metrics

RMSE

Qualitative ?



# 2. Installation
To clone the repository. use `git clone https://github.com/JuanitaSmith/disaster_recovery_pipelines.git`

- Project environment was built using Anaconda.
- Python 3.10 interpreter was used.
- Refer to `requirements.txt` for libraries and versions needed to build your environment.
- Use below commands to rebuild the environment automatically
  - `conda install -r requirements.txt` or 
  - `pip install -r requirements.txt` or 
  - `conda install -c conda-forge --yes --file requirements.txt`
- Refer to `environment.yaml` for environment setup and conda channels used
- Note: Library `iterative-stratification` was installed using pip


# 4. Instructions
Run the following commands in the project's **root directory** to set up your database and model.

A configuration file `src/config.pg` contains defaults for the project structure and file paths. 
Ideally do not change this structure.

**IMPORTANT**: MAKE SURE ALL COMMANDS ARE RUN FROM THE TERMINAL IN THE MAIN PROJECT ROOT

python -m src.preprocessing_data_gathering
python -m src.preprocessing_data_cleaning
python -m src.preprocessing_nlp
python -m unittest tests.test_recommender
python runmovieapp.py  


# 6. Input File Descriptions

2 Datasets:

1) Kaggle 48,000+ movies dataset downloaded [here](https://www.kaggle.com/datasets/yashgupta24/48000-movies-dataset)

This dataset contains the information of 48k+ movies.
which is perfect for content recommendations.
The schema of the dataset:

* Title
* Poster Link
* Genre
* Actors
* Director
* Description
* Date Published
* keyword
* Rating Count
* Best Rating
* Worst Rating
* Rating Value
* Review Date
* Review Author
* Review Body

2) Wikipedia Movie PLots, available on kaggle can be downloaded from [here](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)

Plot descriptions for ~35,000 movies

The dataset contains descriptions of 34,886 movies from around the world. Column descriptions are listed below:

* Release Year - Year in which the movie was released
* Title - Movie title
* Origin/Ethnicity - Origin of movie (i.e. American, Bollywood, Tamil, etc.)
* Director - Director(s)
* Plot - Main actor and actresses
* Genre - Movie Genre(s)
* Wiki Page - URL of the Wikipedia page from which the plot description was scraped
* Plot - Long form description of movie plot (WARNING: May contain spoilers!!!)

3) MovieLens 32M —
   downloaded [here](https://grouplens.org/datasets/movielens/32m/) or [here](https://grouplens.org/datasets/movielens/32m/)
MovieLens 32M movie ratings. 
Stable benchmark dataset. 
32 million ratings and two million tag applications applied to 87,585 movies by 200,948 users. Collected 10/2023 Released 05/2024


4) imdbd - IMDB & TMDB Movie Metadata Big Dataset (>1M) https://www.kaggle.com/datasets/shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m

5) wiki dataset 2 https://www.kaggle.com/datasets/kartikeychauhan/movie-plots

6) wiki dataset 3 https://www.kaggle.com/datasets/exactful/wikipedia-movies?select=2020s-movies.csv

https://www.kaggle.com/datasets/shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m/data


7) imdb ratings and plots for 1500 movies
https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset/code

 8) later for llm poc https://www.kaggle.com/code/francispimentel/imdb-spoiler-detection-exploratory-data-analysis

9 IMDB https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates/data

# 7. Modelling

## Data cleaning

Preparation notebook is stored in `notebooks/ETL Pipeline Preparation.ipynb`



The following data cleaning was completed:
- category 'child_alone' was dropped as it had all entries 'False', giving us no ability to learn
- clean categories so each category appear in its own column with binary indicator 1 or 0
- merge categories and messages datasets using 'id' as key to join
- id was set as a unique index, which is especially needed for language translation during OpenAI
- duplicated messages and indexes were dropped
- column 'original' was dropped as it's unnecessary for the model
- category `related` had 376 records that contained a non-binary value of '2', 
which was changed to '0' as no other categories contained true labels for such records.


## Modelling approach

Preparation notebook is stored in `notebooks/ML Pipeline Preparation.ipynb`
XGBOOST algorythm was used, as it now supports multi-label classification out of the box.

### Dealing with Imbalance
- To deal with imbalance labels, the following techniques were used:
  - Multi-label stratification split into test and train datasets, 
    using package [iterative-stratification](https://pypi.org/project/iterative-stratification/)
  - A new class `src/mloversampler.py` was developed to perform over-sampling 
    for minority classes in MULTI-LABEL classification problems by either duplicating records or using augmentation.
    The Level of duplication is determined by a ratio factor representing the severity of imbalance of each label.
    Example: label 'offer' is duplicated 20 times, whilst label security is duplicated 6 times.
  - A new custom `focal loss` function in class `src/focalloss.py` was developed
    to reduce the importance of the majority class. 
    This function is used as loss function in XGBOOST hyper parameter `eval_metric`.

Result after stratified split and oversampling: 
- Labels are evenly distributed in label datasets before and after stratified split.
- Oversampling is only applied to the training dataset, number of records increased from 17,452 to 30,815 in the notebook preparation.
<img src="disasterapp/static/assets/oversampling_results.png" alt="oversampling"/>

### Cross-validation

During cross-validation, GridSearchCV was used for hyperparameter tuning as it was a project requirement.
Due to long runtimes, the grid search was restricted to `max_depth` and `n_estimators` only.

As a second step,
[OPTUNA](https://www.dailydoseofds.com/bayesian-optimization-for-hyperparameter-tuning/) 
was used for further hyperparameter tuning, as it's much faster.

Both Grid search and Random Search evaluate every hyperparameter configuration independently. 
Thus, they iteratively explore all hyperparameter configurations to find the most optimal one.

However, Bayesian Optimization takes informed steps based on the results of the previous hyperparameter configurations.
This lets it confidently discard non-optimal configurations. 
Consequently, the model converges to an optimal set of hyperparameters much faster.

### Evaluation metrics

**Precision macro** score was used as the main evaluation metrics. 
During a disaster, there are limited resources and services, 
and we want to send resources where we are sure it is necessary. 
Some messages are very vague and unclear.
It's not so easy to get a high precision macro score; it needs extensive engineering effort to get great results.

### Model performance

Model performance during training using **macro precision** as scoring, is increased after grid search.
<img src="disasterapp/static/assets/model_output_results.png" alt="model_output"/>

Final model performance on test data is amazing with 0.84 micro precision and 0.86 macro performance !! 
Great performance on imbalanced labels.
<img src="disasterapp/static/assets/final_model_performance.png" alt="model_output"/>


# 8. Flask Web App

User can input a message, select the genre and click on the button 'Classify Message'.

The saved classification model will be used to classify the message.
All positive classes will be highlighted in green.

<img src="disasterapp/static/assets/website_output.png" alt="website_output"/>


# 9. Classification with OpenAI

Using OpenAI embeddings during classification modeling is showing excellent results during training, 
with a high micro precision of 0.86, 
but poor results for imbalanced labels on the left of the graph with macro precision very low at 0.55.

Surprisingly, `CountVectorizer` and `TfidfTransformer` seems to be winner !!!???

<img src="disasterapp/static/assets/openai_comparison.png" alt="openai"/>



# 9. Skills learned:

Skills applied in this project:

- Web Development using Flask, Plotly and Software Engineering
- Clean and modular code, see custom modules and classes
- GIT version control
- Automated unit testing using library `unittest`, see folder `tests`
- Logging: see folder `logging` for logging results
- Introduction to Object-Oriented Programming - see `src/translator.py` and `mloversampler.py` for custom classes
- Data Engineering: Building pipelines using `scikit-learn` `Pipeline`

# 10. Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Appen for the data.

* Movies dataset (Licence: [CCO: Public Domain](://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots))
* Wikipedia Movie Plots (License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

# 11. References

[Kaggle movies dataset](https://www.kaggle.com/datasets/yashgupta24/48000-movies-dataset/data)
[Wikipedia Movie PLots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
[Building a movie recommender with OpenAI embeddings](https://medium.com/towards-data-science/recreating-andrej-karpathys-weekend-project-a-movie-search-engine-9b270d7a92e4)
https://www.kaggle.com/datasets/kartikeychauhan/movie-plots