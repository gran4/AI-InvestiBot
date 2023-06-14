from numpy import array

#values that do not go from day to day
#EX: earnings comeout every quarter
excluded_values = (
    "earnings dates",
    "earnings diff"
)


def create_sequences(data, num_days):
    #_________________Do totally NOT understand______________________#
    #______________________From Chat Gpt________________________#
    sequences = []
    labels = []
    for i in range(num_days, len(data)):
        sequences.append(data[i-num_days:i, :])
        labels.append(data[i, 0])
    return array(sequences), array(labels)
