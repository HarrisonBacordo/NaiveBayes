import pandas as pd


trainfields = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'spam']
class_freqs = {}
class_counts = {}
attr_freqs = {}
attr_counts = {}


def classify(attr, class_):
    """
    classifies an email as spam or not spam, given the passed in attributes.
    :param attr: attributes of the email
    :param class_: class to get probability on
    :return: the class which holds the highest probability
    """
    class_prior = class_freqs[class_]
    likelihood = 1
    for i, j in zip(attr_freqs[class_], attr):
        if j == 0:
            likelihood *= 1 - i
        else:
            likelihood *= i
    # predictor_prior = 1
    # for i, j, k in zip(attr_freqs[1], attr_freqs[0], attr):
    #     if k == 0:
    #         predictor_prior *= ((1 - i)*class_prior) + ((1 - j) * (1 - class_prior))
    #     else:
    #         predictor_prior *= (i*class_prior) + (j * (1 - class_prior))
    prob = (likelihood * class_prior)
    if class_ == 0:
        return prob
    f_prob = classify(attr, 0)
    if f_prob > prob:
        return 0
    return 1


def evaluate(data):
    """
    evaluates our model given the passed in data
    :param data: emails to classify
    :return: the answers to each email
    """
    answers = list()
    for email in data:
        answers.append(classify(email, 1))
    return answers


def train(dataset):
    """
    Constructs a probability table based on the given dataset
    :param dataset: the dataset to construct the probability table on
    :return: the counts and frequencies of each attribute given the class,
    which represents the probability table
    """
    classes = sort_classes(dataset)
    # initialise all counts to 1 in order to avoid zero occurrence problems
    counts = {1: [1 for _ in range(12)], 0: [1 for _ in range(12)]}
    for label, emails in classes.items():
        for attr in emails:
            counts[label] += attr[:-1]
    freqs = {1: [x / len(classes[1]) for x in counts[1]],
             0: [x / len(classes[0]) for x in counts[0]]}
    return freqs, counts


def sort_classes(data):
    """
    sorts the given data based on its class (spam or not spam)
    :param data: data to sort
    :return: the sorted classes (in a map)
    """
    spam = {}
    for row in data:
        if row[-1] not in spam:
            spam[row[-1]] = []
        spam[row[-1]].append(row)
    return spam


def main(trainfname, testfname):
    global class_freqs, class_counts, attr_freqs, attr_counts
    # convert data to CSVs readable by pandas
    traindata = pd.read_csv(trainfname, names=trainfields, header=None, skipinitialspace=True, sep=' ')
    testdata = pd.read_csv(testfname, skipinitialspace=True, sep=' ', header=None)
    # get class frequencies and class counts
    freqs, counts = traindata.spam.value_counts(normalize=True), traindata.spam.value_counts(normalize=True)
    class_freqs, class_counts = {1: freqs.get_value(1, 1), 0: freqs.get_value(0, 1)}, \
                                {1: counts.get_value(1, 1), 0: counts.get_value(0, 1)}
    # construct probability table on training dataset
    attr_freqs, attr_counts = train(traindata.values)
    # classify the unlabelled data based on the constructed probability table
    classifications = evaluate(testdata.values)
    print(classifications)


if __name__ == '__main__':
    main("ass3DataFiles/part2/spamLabelled.dat", "ass3DataFiles/part2/spamUnlabelled.dat")
