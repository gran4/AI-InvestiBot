from numpy import array

def create_sequences(data, num_days):
    sequences = []
    labels = []
    for i in range(num_days, len(data)):
        sequences.append(data[i-num_days:i, 0])
        labels.append(data[i, 0])
    return array(sequences), array(labels)