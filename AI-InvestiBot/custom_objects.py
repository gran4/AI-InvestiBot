from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Conv2D, GlobalAveragePooling2D, Reshape, BatchNormalization, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss, MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.activations import linear
from tensorflow import sign, reduce_mean
import tensorflow as tf


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
        self.mae_loss = MeanAbsoluteError()

    def call(self, y_true, y_pred):
        mae_loss = self.mae_loss(y_true, y_pred)

        #see if they go the same direction
        direction_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_pred[:-1])))
        #see if the pred going in the more extreme space in directions
        space_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_true[:-1])))
        
        both_over_zero = tf.cast(tf.logical_and(tf.greater(y_true, 0), tf.greater(y_pred, 0)), tf.float32)
        both_under_zero = tf.cast(tf.logical_and(tf.less(y_true, 0), tf.less(y_pred, 0)), tf.float32)
        both_equal_zero = tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 0)), tf.float32)
        #Sees if they are positive of negitive together
        together_loss = both_over_zero + both_under_zero + both_equal_zero

        # Combine the losses with different weights
        combined_loss = together_loss*.1+direction_penalty*.1+mae_loss*.4+space_penalty*.05#0.7 * huber_loss + 0.3 * mse_loss + 0.5 * direction_penalty

        return combined_loss


def create_LSTM_model(shape: Tuple) -> Sequential:
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=(2), kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=shape, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dense(1, activation=linear))

    model.compile(optimizer=Adam(learning_rate=.001), loss=CustomLoss2())
    return model


def create_LSTM_model2(shape: Tuple) -> Sequential:
    model = Sequential()

    # Add Conv2D layers to process the 4D input
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=shape, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=PReLU(), kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))

    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(Reshape(target_shape=(1, -1)))

    # Add LSTM layers to process the flattened sequence
    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(LSTM(units=64, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    # Add the final output layer
    model.add(Dense(units=1, activation='linear'))  # Assuming regression problem

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=.0005, clipvalue=0.1), loss=CustomLoss2())
    return model
