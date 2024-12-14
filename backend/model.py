import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.mixed_precision import DTypePolicy

class MovieLensModel(Model):
    def __init__(self, num_users, num_movies, embedding_dim=32, **kwargs):
        super(MovieLensModel, self).__init__(**kwargs)
        
        # Embedding layers
        self.user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name="user_embedding")
        self.movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim, name="movie_embedding")
        
        # Fully connected layers
        self.dense1 = Dense(64, activation="relu", name="dense1")
        self.dense2 = Dense(32, activation="relu", name="dense2")
        self.output_layer = Dense(1, activation="linear", name="output_layer")

    def call(self, inputs):
        user_input, movie_input = inputs

        # Embed users and movies
        user_vector = self.user_embedding(user_input)
        movie_vector = self.movie_embedding(movie_input)

        # Combine user and movie embeddings
        combined = tf.concat([user_vector, movie_vector], axis=-1)
        combined = Flatten()(combined)

        # Pass through dense layers
        x = self.dense1(combined)
        x = self.dense2(x)
        output = self.output_layer(x)

        return output

    def get_config(self):
        config = super(MovieLensModel, self).get_config()
        config.update({
            "num_users": self.user_embedding.input_dim,
            "num_movies": self.movie_embedding.input_dim,
            "embedding_dim": self.user_embedding.output_dim,
            "dtype_policy": str(DTypePolicy("float32"))  # Explicitly set dtype policy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_users=config["num_users"],
            num_movies=config["num_movies"],
            embedding_dim=config["embedding_dim"]
        )

# Function to save the model
def save_movielens_model(model, path):
    model.save(path, save_format="tf")  # Save in SavedModel format

# Function to load the model
def load_movielens_model(path):
    with custom_object_scope({"MovieLensModel": MovieLensModel, "DTypePolicy": DTypePolicy}):
        return tf.keras.models.load_model(path)

# Example usage
if __name__ == "__main__":
    num_users = 1000
    num_movies = 500
    embedding_dim = 32

    # Create the model
    model = MovieLensModel(num_users=num_users, num_movies=num_movies, embedding_dim=embedding_dim)

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Save the model
    save_path = "movielens/movie_recommender_model"
    save_movielens_model(model, save_path)

    # Load the model
    loaded_model = load_movielens_model(save_path)

    # Print summary
    loaded_model.summary()
