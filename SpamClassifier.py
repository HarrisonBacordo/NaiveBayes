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
    prob = 0
    if class_ == 1:
        return max(prob, classify(attr, False))
    return prob


def evaluate(data):
    """
    evaluates our model given the passed in data
    :param data: emails to classify
    :return: the answers to each email
    """
    answers = list()
    for email in data:
        answers.append(classify(email, 1))
    return 0


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
    return counts, freqs


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
    testdata = pd.read_csv(testfname, skipinitialspace=True)
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
