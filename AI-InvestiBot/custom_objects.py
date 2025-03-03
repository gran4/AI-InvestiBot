from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, BatchNormalization, PReLU, Flatten, Dropout, Bidirectional, TimeDistributed, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss, MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.activations import linear
from tensorflow import sign, reduce_mean
import tensorflow as tf
import numpy as np


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
        self.window_size = 7
        self.focused_training = False

    def call(self, y_true, y_pred):
        mae_loss = self.mae_loss(y_true, y_pred)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        moving_avg_error = 0
        if not self.focused_training:
            y_true2 = tf.expand_dims(y_true, axis=-1)
            y_pred2 = tf.expand_dims(y_pred, axis=-1)
            
            filter_shape = [self.window_size, 1, 1]
            filter_weights = tf.ones(filter_shape) / float(self.window_size)
            
            ma_y_true = tf.nn.conv1d(y_true2, filters=filter_weights, stride=1, padding='SAME')
            ma_y_pred = tf.nn.conv1d(y_pred2, filters=filter_weights, stride=1, padding='SAME')
            
            moving_avg_error = tf.reduce_mean(tf.abs(ma_y_pred - ma_y_true))

        direction_penalty = tf.reduce_mean(tf.abs(tf.tanh(y_true[1:] - y_true[:-1]) - tf.tanh(y_pred[1:] - y_pred[:-1])))
        diff_true = tf.clip_by_value(y_true[1:] - y_true[:-1], -1, 1)
        diff_pred = tf.clip_by_value(y_pred[1:] - y_pred[:-1], -1, 1)
        direction_penalty = tf.reduce_mean(tf.abs(tf.tanh(diff_true) - tf.tanh(diff_pred)))

        space_penalty = tf.reduce_mean(tf.abs(tf.tanh(y_true[1:] - y_true[:-1]) - tf.tanh(y_pred[1:] - y_true[:-1])))
        space_penalty = tf.clip_by_value(space_penalty, 0, 0.5)

        both_over_zero = tf.cast(tf.logical_and(tf.greater(y_true, 0), tf.greater(y_pred, 0)), tf.float32)
        both_under_zero = tf.cast(tf.logical_and(tf.less(y_true, 0), tf.less(y_pred, 0)), tf.float32)
        both_equal_zero = tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 0)), tf.float32)
        together_loss = both_over_zero + both_under_zero + both_equal_zero
        together_loss = tf.clip_by_value(together_loss, 0, 0.5)

        non_existance_penalty = tf.cast(tf.logical_and(tf.greater(y_pred, -0.25), tf.less(y_pred, 0.25)), tf.float32)
        non_existance_penalty2 = tf.cast(tf.logical_and(tf.greater(y_pred, -1), tf.less(y_pred, 1)), tf.float32)
        over_existance_penalty = tf.cast(tf.logical_or(tf.greater(y_pred, 10), tf.less(y_pred, -10)), tf.float32)
        over_existance_penalty = tf.clip_by_value(over_existance_penalty, 0, 0.5)
        over_existance_penalty2 = tf.cast(tf.logical_or(tf.greater(y_pred, 20), tf.less(y_pred, -20)), tf.float32)
        over_existance_penalty2 = tf.clip_by_value(over_existance_penalty2, 0, 0.5)

        change = tf.abs(y_pred[1:] - y_pred[:-1])
        non_change_loss = tf.reduce_mean(tf.cast(tf.less(change, 0.1), tf.float32))
        non_change_loss2 = tf.reduce_mean(tf.cast(tf.less(change, 0.2), tf.float32))
        sign_converted_preds = tf.sign(y_pred)
        zero_centering_loss = tf.abs(tf.reduce_mean(sign_converted_preds))

        one_sided_penalty = tf.reduce_mean(
            tf.abs(tf.tanh(y_true[1:] - y_true[:-1]) - tf.tanh(y_pred[1:] - y_pred[:-1]))
        )
        prediction_variance = tf.math.reduce_variance(y_pred)
        diversity_penalty = 1.0 / (1.0 + prediction_variance)

        combined_loss = (
            together_loss * 0.1 +
            #direction_penalty * 0.05 +
            #one_sided_penalty * 0.05 +
            mae_loss * 0.1 +
            moving_avg_error * 0.002 +
            #space_penalty * 0.02 +
            #non_change_loss * 0.1 +
            #non_change_loss2 * 0.1 +
            zero_centering_loss * 0.15 + 
            diversity_penalty * 0.10
        )

        return combined_loss
    


""" @tf.keras.saving.register_keras_serializable()
class CustomLoss2(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mae_loss = MeanAbsoluteError()
        self.window_size = 7
        self.focused_training = False

    def call(self, y_true, y_pred):
        mae_loss = self.mae_loss(y_true, y_pred)
        #mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        # Ensure predictions and labels have the same shape
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        moving_avg_error = 0
        if not self.focused_training:
            # Calculate the moving average for predictions and actual data
            y_true2 = tf.expand_dims(y_true, axis=-1)  # Shape: [batch_size, sequence_length, 1]
            y_pred2 = tf.expand_dims(y_pred, axis=-1)  # Shape: [batch_size, sequence_length, 1]
            
            # Define the filter for moving average
            filter_shape = [self.window_size, 1, 1]
            filter_weights = tf.ones(filter_shape) / float(self.window_size)
            
            # Compute moving averages using 1D convolution
            ma_y_true = tf.nn.conv1d(y_true2, filters=filter_weights, stride=1, padding='SAME')
            ma_y_pred = tf.nn.conv1d(y_pred2, filters=filter_weights, stride=1, padding='SAME')
            
            # Compute Mean Absolute Error between the moving averages
            moving_avg_error = tf.reduce_mean(tf.abs(ma_y_pred - ma_y_true))


        #see if they go the same direction
        direction_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_pred[:-1])))
        #see if the pred going in the more extreme space in directions
        space_penalty = reduce_mean(abs(sign(y_true[1:] - y_true[:-1]) - sign(y_pred[1:] - y_true[:-1])))
        space_penalty = tf.clip_by_value(space_penalty, 0, .5)

        both_over_zero = tf.cast(tf.logical_and(tf.greater(y_true, 0), tf.greater(y_pred, 0)), tf.float32)
        both_under_zero = tf.cast(tf.logical_and(tf.less(y_true, 0), tf.less(y_pred, 0)), tf.float32)
        both_equal_zero = tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 0)), tf.float32)
        #Sees if they are positive of negitive together
        together_loss = both_over_zero + both_under_zero + both_equal_zero
        together_loss = tf.clip_by_value(together_loss, 0, .5)

        non_existance_penalty = tf.cast(tf.logical_and(tf.greater(y_pred, -.25), tf.less(y_pred, .25)), tf.float32)
        non_existance_penalty2 = tf.cast(tf.logical_and(tf.greater(y_pred, -1), tf.less(y_pred, 1)), tf.float32)

        # Calculate the penalty based on how much the predictions exceed the threshold
        over_existance_penalty = tf.cast(tf.logical_or(tf.greater(y_pred, 10), tf.less(y_pred, -10)), tf.float32)
        over_existance_penalty = tf.clip_by_value(over_existance_penalty, 0, .5)
        over_existance_penalty2 = tf.cast(tf.logical_or(tf.greater(y_pred, 20), tf.less(y_pred, -20)), tf.float32)
        over_existance_penalty2 = tf.clip_by_value(over_existance_penalty2, 0, .5)
        #over_existance_penalty3 = tf.cast(tf.logical_or(tf.greater(y_pred, 20), tf.less(y_pred, -20)), tf.float32)
        #over_existance_penalty3 = tf.clip_by_value(over_existance_penalty3, 0, .5)
        # Combine the losses with different weights

        change = tf.abs(y_pred[1:] - y_pred[:-1])
        non_change_loss = reduce_mean(tf.cast(tf.less(change, 0.1), tf.float32))
        zero_centering_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(y_pred) - 0))
        # spatial loss is going to one side
        # the average % change is different from the actual
        combined_loss = together_loss*.1+direction_penalty*.1+mae_loss*.4+moving_avg_error*0.1+space_penalty*.05+non_change_loss*0.1+zero_centering_loss*0.2#+over_existance_penalty*.2+over_existance_penalty2*.2#+over_existance_penalty3#0.7 * huber_loss + 0.3 * mse_loss + 0.5 * direction_penalty

        return combined_loss """




def create_LSTM_model(input_shape: Tuple, loss: Loss=CustomLoss2()) -> Sequential:
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=(2), kernel_regularizer=tf.keras.regularizers.l2(0.02), input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.02), kernel_initializer='he_normal')))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l1(0.02), kernel_initializer='he_normal')))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation=linear))

    model.compile(optimizer=Adam(learning_rate=.001), loss=loss)
    return model

def percentage_change(x):
    return tf.keras.backend.tanh(x) # Convert raw output to percentage change

def create_LSTM_model2(input_shape: Tuple, loss: Loss=CustomLoss2()) -> Sequential:
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.002), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    # model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    # model.add(PReLU())
    # model.add(Dropout(0.2))

    # model.add(LSTM(32, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    # model.add(PReLU())
    # model.add(Dropout(0.2))


    #  Dense layers for the final prediction
    model.add(Dense(1, activation='linear'))

    #$model.add(Flatten())

    # Add LSTM layers to process the flattened sequence
    #model.add(LSTM(units=64, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))
    #model.add(BatchNormalization())
    #model.add(PReLU())
    # Add the final output layer
    # model.add(Dense(units=1, activation=percentage_change))  # Assuming regression problem

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=.0005, clipvalue=1.0), loss=loss)
    return model


#Add a seperate model for earnings(Only looks at right after earnings)?


    model.add(TimeDistributed((Conv1D(filters=64, kernel_size=(2), input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(PReLU()))

    model.add(TimeDistributed(Reshape((-1, 64))))
    # Merge TimeDistributed output to form a 3D tensor for LSTM
    model.add(Reshape((input_shape[0], -1)))
