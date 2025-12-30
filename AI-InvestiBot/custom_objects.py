from typing import Tuple

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Conv1D,
    Conv2D,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Reshape,
    BatchNormalization,
    PReLU,
    Dropout,
    Multiply,
    Input,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss, MeanSquaredError, Huber, MeanAbsoluteError, BinaryCrossentropy
from tensorflow.keras.activations import linear
from tensorflow import sign, reduce_mean
import tensorflow as tf
import keras.backend as K


@tf.keras.saving.register_keras_serializable()
class DirectionalConsistencyLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_loss = Huber()

    def call(self, y_true, y_pred):
        huber_loss = self.base_loss(y_true, y_pred)

        direction_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_pred[:-1])))
        space_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_true[:-1])))

        combined_loss = huber_loss + 0.2 * direction_penalty + 0.2 * space_penalty
        return combined_loss


@tf.keras.saving.register_keras_serializable()
class ReversalHuberLoss(Loss):
    def __init__(self, threshold: float = 30.0, huber_delta: float = 2.0, amplitude_weight: float = 0.18, variance_weight: float = 0.12, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber = Huber(delta=huber_delta)
        self.amplitude_weight = amplitude_weight
        self.variance_weight = variance_weight

    def call(self, y_true, y_pred):
        huber_loss = self.huber(y_true, y_pred)

        magnitude_penalty = K.mean(K.abs(K.abs(y_true) - K.abs(y_pred)))
        true_std = tf.math.reduce_std(y_true)
        pred_std = tf.math.reduce_std(y_pred)
        variance_penalty = tf.abs(true_std - pred_std)
        direction_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_pred[:-1])))
        space_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_true[:-1])))

        both_over_zero = tf.cast(tf.logical_and(tf.greater(y_true, 0), tf.greater(y_pred, 0)), tf.float32)
        both_under_zero = tf.cast(tf.logical_and(tf.less(y_true, 0), tf.less(y_pred, 0)), tf.float32)
        both_equal_zero = tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 0)), tf.float32)
        together_loss = 1.0 - reduce_mean(both_over_zero + both_under_zero + both_equal_zero)

        combined_loss = (
            huber_loss * 0.4
            + self.amplitude_weight * magnitude_penalty
            + self.variance_weight * variance_penalty
            + direction_penalty * 0.18
            + space_penalty * 0.1
            + together_loss * 0.04
        )

        return combined_loss


def create_LSTM_model(shape: Tuple) -> Sequential:
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=(2), kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=shape, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal', recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal', recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation=linear))

    model.compile(optimizer=Adam(learning_rate=.001), loss=Huber())
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
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.3))
    # Add the final output layer
    model.add(Dense(units=1, activation='linear'))  # Assuming regression problem

    # Compile the model with the reversal-aware custom loss for richer signals
    model.compile(optimizer=Adam(learning_rate=.0005, clipvalue=0.1), loss=ReversalHuberLoss())
    return model


def create_lightweight_model(shape: Tuple) -> Sequential:
    """
    Simpler single-layer LSTM that is faster to train and less prone to overfitting.
    Useful when experimenting with new indicators or limited datasets.
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=shape, kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
    return model


class HeteroscedasticLoss(Loss):
    def __init__(self, min_log_var: float = -10.0, max_log_var: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def call(self, y_true, y_pred):
        mean = y_pred[..., 0]
        log_var = tf.clip_by_value(y_pred[..., 1], self.min_log_var, self.max_log_var)
        precision = tf.exp(-log_var)
        loss = precision * tf.square(y_true - mean) + log_var
        return tf.reduce_mean(loss)


def create_context_gated_model(shape: Tuple) -> Model:
    inputs = Input(shape=shape)
    x = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(inputs)
    x = LSTM(48, return_sequences=True, kernel_initializer='he_normal')(x)
    x = LSTM(32, kernel_initializer='he_normal')(x)

    context = GlobalAveragePooling1D()(inputs)
    gate = Dense(32, activation='sigmoid')(context)
    gated = Multiply()([x, gate])
    gated = Dropout(0.2)(gated)

    output = Dense(1, activation='linear')(gated)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
    return model


def create_probabilistic_model(shape: Tuple) -> Model:
    inputs = Input(shape=shape)
    x = LSTM(64, return_sequences=True, kernel_initializer='he_normal')(inputs)
    x = LSTM(32, kernel_initializer='he_normal')(x)
    x = Dropout(0.2)(x)
    mean = Dense(1)(x)
    log_var = Dense(1)(x)
    outputs = Concatenate()([mean, log_var])

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=HeteroscedasticLoss())
    return model


def create_directional_model(shape: Tuple) -> Sequential:
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same', input_shape=shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(LSTM(32, kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy())
    return model


class FocalLoss(Loss):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma) * y_true + (1 - self.alpha) * tf.pow(y_pred, self.gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)


class BalancedFocalLoss(FocalLoss):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        lambda_balance: float = 5.0,
        target_mean: float = 0.5,
        tolerance: float = 0.1,
        mean_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(gamma=gamma, alpha=alpha, **kwargs)
        self.lambda_balance = lambda_balance
        self.target_mean = target_mean
        self.tolerance = tolerance
        self.mean_weight = mean_weight

    def call(self, y_true, y_pred):
        base_loss = super().call(y_true, y_pred)
        mean_pred = tf.reduce_mean(y_pred)
        deviation = tf.abs(mean_pred - self.target_mean)
        penalty = tf.where(
            deviation > self.tolerance,
            tf.square(deviation - self.tolerance),
            0.0,
        )
        variance = tf.reduce_mean(tf.square(y_pred - mean_pred))
        anchor = self.mean_weight * (mean_pred - self.target_mean)
        return base_loss + self.lambda_balance * (penalty + 0.5 * variance) + anchor


def create_directional_model_focal(shape: Tuple) -> Sequential:
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same', input_shape=shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(LSTM(32, kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    focal_loss = BalancedFocalLoss(
        gamma=1.5,
        alpha=0.65,
        lambda_balance=5.0,
        target_mean=0.5,
        tolerance=0.1,
        mean_weight=0.05,
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss=focal_loss, metrics=['accuracy'])
    return model
