from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from model import MovieLensModel
import tensorflow as tf
from flask_cors import CORS

def load_movie_lens_model(model_path, num_users, num_movies, user_ids=None, movie_ids=None):
    # Create a new instance of MovieLensModel with the required parameters
    model = MovieLensModel(num_users=num_users, num_movies=num_movies, user_ids=user_ids, movie_ids=movie_ids)
    model.load_weights(model_path)  # Load the pre-trained model weights
    return model

# Load the movie dataset
ratings_df = pd.read_csv('movielens/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
users_df = pd.read_csv('movielens/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
movies_df = pd.read_csv(
    'movielens/u.item',
    sep='|',
    encoding='latin-1',
    names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] +
          ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
           'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
           'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
)

# Preprocessing data
ratings_df['user_id'] = ratings_df['user_id'].astype(str)
ratings_df['movie_id'] = ratings_df['movie_id'].astype(str)
users_df['user_id'] = users_df['user_id'].astype(str)
movies_df['movie_id'] = movies_df['movie_id'].astype(str)

# Extract unique user_ids and movie_ids
user_ids = ratings_df["user_id"].unique().tolist()
movie_ids = ratings_df["movie_id"].unique().tolist()

# Instantiate the model
num_users = len(user_ids) + 1  # Get actual number of unique users + 1 for padding
num_movies = len(movie_ids) + 1

# Load your trained model
model = load_movie_lens_model('movielens/movie_recommender_model.keras', num_users, num_movies, user_ids, movie_ids)

# Movie ID to name mapping
movie_id_to_name = dict(zip(movies_df['movie_id'].astype(str), movies_df['title']))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the Movie Recommender API!"

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content, no need for a favicon

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_name = request.args.get('movie_name')
    user_id = request.args.get('user_id', default='user1')  # Use 'user1' as default if no user_id is provided
    recommendations = get_recommendations_by_movie_name(movie_name, user_id)
    if not recommendations:
        return jsonify({"recommendations": []})  # Send an empty list if no recommendations
    return jsonify({"recommendations": recommendations})
  
def get_recommendations_by_movie_name(movie_name, user_id, top_n=5):
    try:
        # Get movie ID from the dataset based on the movie name
        movie_id = movies_df[movies_df['title'] == movie_name]['movie_id'].values[0]
    except IndexError:
        return "Movie not found! Try another one."

    # Check if user has rated the movie
    user_rating = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['movie_id'] == movie_id)]

    # If the user hasn't rated the movie, assume they like it (assign a high rating like 5)
    if user_rating.empty:
        assumed_rating = 5
    else:
        assumed_rating = user_rating['rating'].values[0]
    
    # Create a copy of ratings_df to avoid modifying the global variable
    ratings_df_copy = ratings_df.copy()

    # Add the assumed rating for the movie if the user hasn't rated it
    # This assumes the user rates the selected movie highly (like 5)
    ratings_df_copy = pd.concat([ratings_df_copy, pd.DataFrame({
        'user_id': [user_id],
        'movie_id': [movie_id],
        'rating': [assumed_rating]
    })], ignore_index=True)
    
    # Create a user-item interaction matrix
    user_movie_matrix = ratings_df_copy.pivot_table(index='user_id', columns='movie_id', values='rating')

    # Fill NaNs with 0 (or you can use some other strategy like mean imputation)
    user_movie_matrix = user_movie_matrix.fillna(0)

    # Calculate the cosine similarity between the selected movie and all other movies
    cosine_similarities = cosine_similarity(user_movie_matrix.T)  # Transpose to compare movies with each other

    # Get index of the selected movie
    movie_index = user_movie_matrix.columns.get_loc(movie_id)

    # Get similarities for the selected movie
    similarities = cosine_similarities[movie_index]

    # Get the top N most similar movies
    similar_movie_indices = similarities.argsort()[-(top_n + 1):-1]  # Exclude the movie itself
    similar_movie_ids = user_movie_matrix.columns[similar_movie_indices]

    # Get movie names from movie_id
    similar_movie_names = movies_df[movies_df['movie_id'].isin(similar_movie_ids)]['title'].tolist()
    return similar_movie_names

if __name__ == '__main__':
    app.run(debug=True)
