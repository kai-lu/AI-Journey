"""
https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
https://pythonprogramminglanguage.com/naive-bayes-classifier/
"""
# Example of calculating class probabilities
from math import sqrt
from math import pi
from math import exp
from tabulate import tabulate  # to print lists in a pretty form
from sklearn.naive_bayes import GaussianNB


# Split the dataset by class values, returns a dictionary
def separate_by_class(_dataset):
    separated = dict()
    for sample_index in range(len(_dataset)):
        vector = _dataset[sample_index]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    print(f"\n separate dataset by class:")
    for key in separated:
        separated_list = separated[key]
        print(f"class value {key}\n sample data {separated_list}")
    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(_dataset):
    # print(f'zip(*_dataset):{list(zip(*_dataset))}')
    _summaries = [(mean(column), stdev(column), len(column)) for column in zip(*_dataset)]
    del (_summaries[-1])
    return _summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(_dataset):
    separated = separate_by_class(_dataset)
    _summaries = dict()
    for class_value, rows in separated.items():
        _summaries[class_value] = summarize_dataset(rows)
    print(f"\n variable _summaries in summarize_by_class is")
    for key in _summaries:
        _summaries_list = _summaries[key]
        print(f"class value {key}\n data mean and standard deviation number of samples for columns\n {_summaries_list}")
    return _summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, _mean, _stdev):
    exponent = exp(-((x - _mean) ** 2 / (2 * _stdev ** 2)))
    return (1 / (sqrt(2 * pi) * _stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(_summaries, test_data):
    total_rows = sum([_summaries[label][0][2] for label in _summaries])
    _probabilities = dict()
    for class_value, class_summaries in _summaries.items():
        _probabilities[class_value] = _summaries[class_value][0][2] / float(total_rows)
        for attr_index in range(len(class_summaries)):
            _mean, _stdev, _ = class_summaries[attr_index]
            _probabilities[class_value] *= calculate_probability(test_data[attr_index], _mean, _stdev)
    return _probabilities


# Test calculating class probabilities
data_train = [[3.393533211, 2.331273381, 0],
              [3.110073483, 1.781539638, 0],
              [1.343808831, 3.368360954, 0],
              [3.582294042, 4.67917911, 0],
              [2.280362439, 2.866990263, 0],
              [7.423436942, 4.696522875, 1],
              [5.745051997, 3.533989803, 1],
              [9.172168622, 2.511101045, 1],
              [7.792783481, 3.424088941, 1],
              [7.939820817, 0.791637231, 1],
              [6.939820817, 1.791637231, 1]]
data_test = [7.423436942, 4.696522875]
print(f'Training data set is:\n{tabulate(data_train)}')
summaries = summarize_by_class(data_train)
probabilities = calculate_class_probabilities(summaries, data_test)
print('\n', probabilities)
predict_class = list(probabilities.values()).index(max(probabilities.values()))
print(f'The class of the test sample is: {predict_class}')

# another approach for this task
data_train_skl_format = [sublist[0:-1] for sublist in data_train]
class_skl_format = [sublist[-1] for sublist in data_train]
# create naive bayes classifier
gaussian_nb = GaussianNB()
# train classifier with dataset
gaussian_nb = gaussian_nb.fit(data_train_skl_format, class_skl_format)
# predict using classifier
prediction = gaussian_nb.predict([data_test])
print(f'The class of the test sample using sklearn is: {prediction}')
