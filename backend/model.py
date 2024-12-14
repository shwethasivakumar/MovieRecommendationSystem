from tensorflow.keras.mixed_precision import Policy
from keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, StringLookup

# Set precision policy
# Set the policy (example)
policy = Policy('float16')
mixed_precision.set_global_policy(policy)

@register_keras_serializable()
class MovieLensModel(tf.keras.Model):
    def __init__(self, num_users, num_movies, user_ids=None, movie_ids=None, embedding_dim=32, **kwargs):
        super(MovieLensModel, self).__init__(**kwargs)
        if user_ids is None or movie_ids is None:
            raise ValueError("user_ids and movie_ids must be provided")
        self.user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)
        self.movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim)
        self.dense = Dense(1)
        self.user_lookup = StringLookup(vocabulary=user_ids, mask_token=None, num_oov_indices=0)
        self.movie_lookup = StringLookup(vocabulary=movie_ids, mask_token=None, num_oov_indices=0)

    def call(self, inputs):
        user_id = self.user_lookup(tf.as_string(inputs["user_id"]))
        movie_id = self.movie_lookup(tf.as_string(inputs["movie_id"]))
        user_vector = self.user_embedding(user_id)
        movie_vector = self.movie_embedding(movie_id)
        return self.dense(user_vector * movie_vector)

    def get_config(self):
        config = super(MovieLensModel, self).get_config()
        config.update({
            'num_users': self.user_embedding.input_dim,
            'num_movies': self.movie_embedding.input_dim,
            'embedding_dim': self.user_embedding.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Function to load model
def load_movie_lens_model(model_path, num_users, num_movies, user_ids, movie_ids):
    model = tf.keras.models.load_model(model_path, custom_objects={'MovieLensModel': MovieLensModel})
    return model
