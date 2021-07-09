import numpy as np
import tensorflow as tf


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class SpeakerEncoder(tf.keras.Model):
    def __init__(self, name="SpeakerEncoder", **kwargs):
        super(SpeakerEncoder, self).__init__(name=name, **kwargs)
        # self.inp = tf.keras.layers.Input(shape=(100, 40), name='partial_utterance')
        self.lstm1 = tf.keras.layers.LSTM(units=40, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(units=256, return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(units=256)
        self.linear = Linear(units=256, input_dim=256)
        self.relu = tf.keras.layers.ReLU()
        self.embed = tf.keras.layers.Lambda(lambda y: y / (tf.norm(y, axis=1, keepdims=True) + 1e-05))

        self.similarity_weight = 0.01
        self.similarity_bias = 0.01

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def call(self, input):
        # x = self.inp(input)
        x = input
        x = self.lstm3(self.lstm2(self.lstm1(x)))
        x = self.relu(self.linear(x))
        # x = self.embed(x)
        return x

    def similarity_matrix(embeds):
        def similarity_matrix_numpy(embeds):
            speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
            centroid_inc = tf.reduce_mean(input_tensor=embeds, axis=1, keepdims=True)
            centroid_inc = tf.divide(centroid_inc, tf.norm(centroid_inc, axis=2, keepdims=True) + 1e-05)

            centroid_exc = (tf.reduce_sum(input_tensor=embeds, axis=1, keepdims=True) - embeds)
            centroid_exc /= (utterances_per_speaker - 1)
            centroid_exc = tf.divide(centroid_exc, (tf.norm(centroid_exc, axis=2, keepdims=True) + 1e-05))

            # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
            # product of these vectors (which is just an element-wise multiplication reduced by a sum).
            # We vectorize the computation for efficiency.
            sim_matrix = np.zeros(shape=(speakers_per_batch, utterances_per_speaker, speakers_per_batch))
            mask_matrix = 1 - tf.eye(speakers_per_batch, dtype=tf.int16)
            for j in range(speakers_per_batch):
                mask = tf.where(mask_matrix[j])[0][0]
                sim_matrix[mask, :, j] = tf.reduce_sum((embeds[mask] * centroid_inc[j]), axis=1)
                sim_matrix[j, :, j] = tf.reduce_sum((embeds[j] * centroid_exc[j]), axis=1)

            sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
            return sim_matrix

        out = tf.numpy_function(func=similarity_matrix_numpy,
                                inp=[embeds], Tout=[tf.float32])
        return out

    def loss():
        speakers_per_batch, utterances_per_speaker = embed.shape[:2]

        # Loss
        sim_matrix = self.similarity_matrix(embed, speaker_label)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker,
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        loss = self.loss_fn(sim_matrix, ground_truth)

        return loss