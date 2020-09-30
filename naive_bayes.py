#Michael Rao
#1001558150

import os
import sys
from math import pi
from math import exp

#Load file
def load_file(file_name):
    dataset = list()
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            dataset.append(line.split())
    return dataset

# Convert string column to float
def create_floats(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Calculate mean 
def mean(list_numbers):
    return sum(list_numbers)/float(len(list_numbers))

# Calculate standard deviation
def standard_dev(list_numbers):
    avg = mean(list_numbers)
    
    
    variance = sum([(x - avg)**2 for x in list_numbers]) / (len(list_numbers) -1)
    ret_val = variance**0.5
    if ret_val == 0:
        return 0.01
    return  ret_val

# Calculate Gaussian Prob
def gaus_prob(x, mean, std_dev):
    exponent = exp(-((x - mean)**2 / (2 * std_dev**2 )))
    return (1 / (((2* pi)**0.5) * std_dev)) * exponent

def class_predicition(data_summary, row):
    total_rows = sum([data_summary[label][0][2] for label in data_summary])
    probs = dict()

    for values, stats in data_summary.items():
        probs[values] = data_summary[values][0][2]/float(total_rows)
        for i in range(len(stats)):
            mean, std_dev, _= stats[i]
            probs[values] *= gaus_prob(row[i], mean, std_dev)
    return probs

# Class prediction for given attribute
def predict(data_summary, test_rows):
    probs = {}
    class_probs = {}
    total_rows = sum([data_summary[label][0][2] for label in data_summary])

    for key in data_summary:
        class_probs[key] = len(data_summary[key])/total_rows

    final = {}
    for row in range(len(test_rows)):
        probs[row] = class_predicition(data_summary,test_rows[row])

    tot_acc = 0.0
    for k in probs:
        prob_x = 0
        acc = 0.0
        for i in range(len(probs[k])):
            prob_x = prob_x + (probs[k][i+1] * class_probs[i+1])
        #print(prob_x)
        prob_y = {}
        for x in range(len(probs[k])):
            prob_y[x+1] = (probs[k][x+1] * class_probs[x+1])/prob_x
        if(max(prob_y,key=prob_y.get) == int(test_rows[k][8])):
            acc = 1
            tot_acc += 1
        print("ID= {0:5d}, predicted= {1:3d}, probability= {2:.4f}, true= {3:3d}, accuracy= {4:4.2f}".format(k + 1,max(prob_y,key=prob_y.get),prob_y[max(prob_y,key=prob_y.get)],int(test_rows[k][8]), acc))
    tot_acc = tot_acc/(k+1)
    print("classification accuracy={0:6.4f}".format(tot_acc))
    

# Create a dictionary of class and values
def class_dictionary(dataset):
    class_dict = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_val = int(vector[-1])
        if (class_val not in class_dict):
            class_dict[class_val] = list()
        class_dict[class_val].append(vector)
    return class_dict 

# Calculate stats for each class
def class_stats(dataset):
    class_dict = class_dictionary(dataset)
    class_summary = dict()
    for value, rows in class_dict.items():
        class_summary[value] = dataset_stats(rows)
    return class_summary

# Create stats for each class
def dataset_stats(dataset):
    data_stats = [(mean(column), standard_dev(column), len(column)) for column in zip(*dataset)]
    del(data_stats[-1])
    return data_stats

# Print training phase
def print_training(train_summary):
    for key in sorted(train_summary.keys()):
        for i in range(len(train_summary[key])):
            print("Class {0}, attribute {1}, mean = {2:.2f}, std = {3:.2f} ".format(key,i + 1,train_summary[key][i][0],train_summary[key][i][1]))


def naive_bayes():
    if(len(sys.argv) < 3):
        print("Insufficient command line args")
        exit()

    filename = sys.argv[1]
    test_file = sys.argv[2]

    dataset = load_file(filename)
    test_set = load_file(sys.argv[2])

    print(dataset)
    for i in range(len(dataset[0])):
        create_floats(dataset, i)

    training_stats = class_stats(dataset)
    print_training(training_stats)

    for i in range(len(test_set[0])):
        create_floats(test_set, i)


    predict(training_stats, test_set)
if __name__ == '__main__':
    naive_bayes()
