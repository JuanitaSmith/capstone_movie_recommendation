from movierecommenderapp import app
from flask import render_template, request
from src import path_imdb_clean, path_ratings_clean, Recommender, \
    path_tags_clean

r = Recommender()

data_loaded = False

@app.before_request
def firstRun():
    """
    Methods to call only once when the webapp is initiated

    - Loads data
    - Create user-item matrix
    - Create total ratings per user
    """
    global data_loaded
    if not data_loaded:
        print('Loading initial data on first run')
        r.logger.info('Loading initial data on first run')
        r.get_data(path_imdb_clean, path_ratings_clean, path_tags_clean)
        r.load_tfidf_vectorizer()
        r.load_user_item_matrix()
        r.count_user_ratings()
        data_loaded = True

# main webpage receiving user input
@app.route('/')
@app.route('/index')
def index():

    return render_template(
        'input.html',
    )

# web page that receives user input and make recommendations
@app.route('/recommender', methods=['POST'])
def recommender():

    keywords = []

    # get user id from form
    # If no user id supplied, default to 0
    try:
        inputUserId = int(request.form.get('inputUserId'))
        if not inputUserId:
            inputUserId = request.args.get('inputUserId')
    except Exception as e:
        inputUserId = 0

    if inputUserId == None:
        inputUserId = 'New'

    print('User id received is {}'.format(inputUserId))
    r.logger.info('User id received is {}'.format(inputUserId))

    # get search text
    search_text = request.form.get('search_input')
    print('Search text received is {}'.format(search_text))
    r.logger.info('Search text received is {}'.format(search_text))

    # Decide which type of recommendation to do:
    # 1) If a search text is entered, do a content-based recommendation
    # 2) If interactions for user id 2 exist, do collaborative filtering
    # 3) If the user is a new user without interactions,
    # do ranked-based recommendation

    movies = {}
    if search_text:
        text = 'Content-based recommendation triggered with text {}'
        r.logger.info(text.format(search_text))
        movies = r.make_content_recommendations(
            tfidf_df=r.tfidf_df,
            tfidf_vectorizer=r.tfidf_vectorizer,
            input_search_text=search_text,
            user_id=inputUserId,
            top_n=20
        )
        if len(movies) == 0:
            recommender_comment =(
                "No movies found for search '{}'".format(search_text))
        else:
            recommender_comment = (
                "Content based recommendation using '{}'".format(search_text))
    else:
        user_exist = (r.df_ratings['user_id'] == inputUserId).any()
        if user_exist:
            print('Interactions for user {} exists'.format(inputUserId))
            r.logger.info(
                'Interactions for user {} exists'.format(inputUserId))

            text = 'Collaborative filtering recommendation was triggered'
            r.logger.info(text)

            movies, search_text = r.user_user_recommendations(
                user_id=inputUserId,
                top_n=20
            )
            if search_text:
                recommender_comment = (
                    "Content based search using '{}'".format(search_text))
            else:
                # keywords = r.get_user_interests(
                #     user_id=inputUserId,
                #     top_n=15)
                recommender_comment = (
                    'What users with similar tastes are watching')
        else:
            print('No Interactions for user {} exists'.format(inputUserId))
            r.logger.info(
                'No Interactions for user {} exists'.format(inputUserId))
            text = 'Ranked-based recommendation was triggered'
            r.logger.info(text)
            movies = r.get_top_movies(n=20)
            recommender_comment = ("As it's your first visit, "
                                   "we recommend these most popular movies")

    nav_texts = " - Recommendations for UserID"

    return render_template(
        'recommender.html',
        inputUserId=inputUserId,
        recommender_comment=recommender_comment,
        nav_texts=nav_texts,
        movies=movies,
        keywords=keywords)



@app.route('/view', methods=['POST'])
def view():
    print('View button is clicked')
    r.logger.info('View button is clicked')

    content = None

    inputUserId = request.args.get('inputUserId')
    print('user id received is {}'.format(inputUserId))

    movie_id = int(request.form.get('view_button'))
    print('movie id view requested: {}'.format(movie_id))

    if movie_id:
        cols = ['doc_full_name', 'doc_description', 'doc_body']
        content = r.df_movies[r.df_movies.index == movie_id]
        content = content.reset_index(drop=False).to_dict(orient='index')[0]

    return render_template(
        'display_movie.html',
        content=content)
