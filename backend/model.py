@register_keras_serializable()
class MovieLensModel(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_dim=32, **kwargs):
        super(MovieLensModel, self).__init__(**kwargs)
        
        # Define the embedding layers for users and movies
        self.user_embedding = tf.keras.layers.Embedding(input_dim=num_users, output_dim=embedding_dim)
        self.movie_embedding = tf.keras.layers.Embedding(input_dim=num_movies, output_dim=embedding_dim)

        # Define a dense layer for output prediction
        self.dense = tf.keras.layers.Dense(1)

        # StringLookup layers for user and movie IDs
        self.user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None, num_oov_indices=0)
        self.movie_lookup = tf.keras.layers.StringLookup(vocabulary=movie_ids, mask_token=None, num_oov_indices=0)

    def call(self, inputs):
        # Convert string IDs to integer indices using StringLookup
        user_id = self.user_lookup(tf.as_string(inputs["user_id"]))
        movie_id = self.movie_lookup(tf.as_string(inputs["movie_id"]))

        # Get embedding vectors for user and movie
        user_vector = self.user_embedding(user_id)
        movie_vector = self.movie_embedding(movie_id)

        # Dot product of user and movie vectors, then pass through dense layer
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
