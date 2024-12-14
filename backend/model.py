import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class MovieLensModel(tf.keras.Model):
    def __init__(self, num_users, num_movies, user_ids=None, movie_ids=None, embedding_dim=32, **kwargs):
        super(MovieLensModel, self).__init__(**kwargs)
        
        if user_ids is None or movie_ids is None:
            raise ValueError("user_ids and movie_ids must be provided")

        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim

        # Define the embedding layers
        self.user_embedding = tf.keras.layers.Embedding(input_dim=num_users, output_dim=embedding_dim)
        self.movie_embedding = tf.keras.layers.Embedding(input_dim=num_movies, output_dim=embedding_dim)
        
        # Dense layer for prediction
        self.dense = tf.keras.layers.Dense(1)

        # StringLookup layers for converting strings to indices
        self.user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None, num_oov_indices=1)
        self.movie_lookup = tf.keras.layers.StringLookup(vocabulary=movie_ids, mask_token=None, num_oov_indices=1)

    def call(self, inputs):
        user_id = self.user_lookup(tf.as_string(inputs["user_id"]))
        movie_id = self.movie_lookup(tf.as_string(inputs["movie_id"]))
        
        user_vector = self.user_embedding(user_id)
        movie_vector = self.movie_embedding(movie_id)
        
        # Predict rating as the dot product of the user and movie embeddings
        return self.dense(user_vector * movie_vector)

    def get_config(self):
        # Ensure we return all necessary configurations for serialization
        config = super(MovieLensModel, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'embedding_dim': self.embedding_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
