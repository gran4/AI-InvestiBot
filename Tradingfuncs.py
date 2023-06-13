from numpy import array

def create_sequences(data, num_days):
    sequences = []
    labels = []
    for i in range(num_days, len(data)):
        sequences.append(data[i-num_days:i, 0])
        labels.append(data[i, 0])
    return array(sequences), array(labels)
import numpy as np

def create_sequences(data, num_days):
    sequences = []
    labels = []
    num_features = data.shape[1]  # Number of features in the data

    for i in range(num_days, len(data)):
        sequence = data[i - num_days:i, :]  # Extract the sequence for all features
        sequences.append(sequence)
        labels.append(data[i, :])

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels
