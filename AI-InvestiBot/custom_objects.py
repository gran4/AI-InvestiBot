from typing import Tuple

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, GRU, Conv2D, Flatten, GlobalAveragePooling2D, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import Loss, MeanSquaredError, Huber
from tensorflow.keras.activations import relu, linear
from tensorflow import sign, reduce_mean
import tensorflow as tf

class PReLU(tf.keras.layers.Layer):
    def __init__(self, alpha_initializer='zeros', **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=input_shape[-1:],
            initializer=self.alpha_initializer,
            trainable=True
        )
        super(PReLU, self).build(input_shape)

    def call(self, inputs):
        return tf.maximum(0.0, inputs) + self.alpha * tf.minimum(0.0, inputs)


@tf.keras.saving.register_keras_serializable()
class CustomLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.huber_loss = Huber()
        self.mse_loss = MeanSquaredError()

    def call(self, y_true, y_pred):
        huber_loss = self.huber_loss(y_true, y_pred)
        mse_loss = self.mse_loss(y_true, y_pred)

        # Calculate the directional penalty
        direction_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_pred[:-1])))
        space_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_true[:-1])))
        direction_penalty -= .12
        space_penalty -= .12
        if direction_penalty < 0:
            return mse_loss-direction_penalty
        if space_penalty < 0:
            return mse_loss-space_penalty

        # Combine the losses with different weights
        combined_loss = direction_penalty*.1+mse_loss+space_penalty*.1#0.7 * huber_loss + 0.3 * mse_loss + 0.5 * direction_penalty

        return combined_loss


@tf.keras.saving.register_keras_serializable()
class CustomLoss2(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.huber_loss = Huber()
        self.mse_loss = MeanSquaredError()

    def call(self, y_true, y_pred):
        huber_loss = self.huber_loss(y_true, y_pred)
        mse_loss = self.mse_loss(y_true, y_pred)

        # Calculate the directional penalty
        direction_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_pred[:-1])))
        space_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_true[:-1])))
        #direction_penalty -= .1
        #space_penalty -= .1
        #if direction_penalty < 0:
        #    return mse_loss-direction_penalty
        #if space_penalty < 0:
        #    return mse_loss-space_penalty

        # Combine the losses with different weights
        combined_loss = direction_penalty*.1+mse_loss*.1+space_penalty*.05#0.7 * huber_loss + 0.3 * mse_loss + 0.5 * direction_penalty

        return combined_loss


def create_LSTM_model(shape: Tuple) -> Sequential:
    model = Sequential()
    model.add(LSTM(16, return_sequences=True, input_shape=shape))
    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(1, activation=linear))

    model.compile(optimizer=Adam(learning_rate=.001), loss=Huber())
    return model


def create_LSTM_model2(shape: Tuple) -> Sequential:
    model = Sequential()

    # Add Conv2D layers to process the 4D input
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation=PReLU(), input_shape=shape, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation=PReLU(), kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))

    # Flatten the output of Conv2D layers
    #model.add(Flatten())

    model.add(LeakyReLU(alpha=0.5))
    model.add(GlobalAveragePooling2D())
    model.add(LeakyReLU(alpha=0.5))
    model.add(Reshape(target_shape=(1, -1)))
    model.add(LeakyReLU(alpha=0.5))

    # Add LSTM layers to process the flattened sequence
    model.add(LSTM(units=16, return_sequences=True, activation=PReLU(), kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(LSTM(units=16, activation=PReLU(), kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.5))
    # Add the final output layer
    model.add(Dense(units=1, activation='linear'))  # Assuming regression problem

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=.001, clipvalue=1.0), loss=CustomLoss2())
    return model
