import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

INDENT = "    "  # for correct python formatting
MAX_DEPTH = 6  # change if overfitting
LEAST_DATA_POINTS = 10

AGGRESSIVE = "2"
NON_AGGRESSIVE = "1"


def read_traffic_data(parent_dir):
    """
    reads all .csv files in a given directory and its subdirectories
    :param parent_dir: directory that has .csv files
    :return: Pandas DataFrame of all .csv traffic data in the directory
    """
    csv_folder = Path(parent_dir)
    df = pd.concat(pd.read_csv(p) for p in csv_folder.glob('**/*.csv'))
    return df


# def check_attributes_validity(traffic_data):
#     columns = list(traffic_data.columns.values)
#     columns.remove('INTENT')
#     for col in columns:
#         least_badness = -1
#         best_threshold = None
#         thresholds = traffic_data[col].unique()
#         thresholds.sort()
#         for threshold in thresholds:
#             # if target is > threshold
#             false_alarm_gt = len(traffic_data.loc[
#                                      (traffic_data[col] > threshold) & (traffic_data['INTENT'] < 2)
#                                      ])
#
#             false_negative_gt = len(traffic_data.loc[
#                                         (traffic_data[col] <= threshold) & (traffic_data['INTENT'] == 2)
#                                         ])
#
#             # if target is <= threshold
#             false_alarm_lt = len(traffic_data.loc[
#                                      (traffic_data[col] <= threshold) & (traffic_data['INTENT'] < 2)
#                                      ])
#
#             false_negative_lt = len(traffic_data.loc[
#                                         (traffic_data[col] > threshold) & (traffic_data['INTENT'] == 2)
#                                         ])
#
#             current_badness_gt = false_alarm_gt + false_negative_gt
#             current_badness_lt = false_alarm_lt + false_negative_lt
#             if current_badness_gt < current_badness_lt:
#                 if least_badness == -1 or current_badness_gt < least_badness:
#                     least_badness = current_badness_gt
#                     best_threshold = threshold
#             else:
#                 if least_badness == -1 or current_badness_lt < least_badness:
#                     least_badness = current_badness_lt
#                     best_threshold = threshold
#         print(f"Attribute {col}: {least_badness} and threshold {best_threshold}")

def EDA_graphs(traffic_data):
    """
    provides information on the correlation between attributes by creating
    graphs of each attribute with every other attribute
    :param traffic_data: the dataframe containing all traffic data
    """
    columns = list(traffic_data.columns.values)
    for i in range(len(columns) - 1):
        for j in range(i + 1, len(columns) - 1):
            x_attribute = columns[i]
            y_attribute = columns[j]
            plt.xlabel(x_attribute)
            plt.ylabel(y_attribute)
            x_vals_agg = traffic_data[x_attribute].loc[traffic_data['INTENT'] == 2].to_numpy(dtype=np.float64)
            y_vals_agg = traffic_data[y_attribute].loc[traffic_data['INTENT'] == 2].to_numpy(dtype=np.float64)
            x_vals_non_agg = traffic_data[x_attribute].loc[traffic_data['INTENT'] < 2].to_numpy(dtype=np.float64)
            y_vals_non_agg = traffic_data[x_attribute].loc[traffic_data['INTENT'] < 2].to_numpy(dtype=np.float64)

            for point in range(len(x_vals_agg)):
                x_vals_agg[point] += random.uniform(-0.25, 0.25)
                y_vals_agg[point] += random.uniform(-0.25, 0.25)
            for point in range(len(x_vals_non_agg)):
                x_vals_non_agg[point] += random.uniform(-0.25, 0.25)
                y_vals_non_agg[point] += random.uniform(-0.25, 0.25)

            plt.scatter(x_vals_agg, y_vals_agg, color='red', s=5)
            plt.scatter(x_vals_non_agg, y_vals_non_agg, color='blue', s=5)
            plt.savefig(f"graphs/{x_attribute}_{y_attribute}.png")
            plt.clf()


def get_best_split(traffic_data, usable_attributes):
    """
    uses the information gain of each possible split to determine the best split to take
    :param traffic_data: the dataframe containing all traffic data
    :param usable_attributes: the unused available attributes
    :return: the best split in the format (attribute, threshold)
    """
    best_gain = (None, '', -1)  # (gain, attribute, threshold)
    for col in usable_attributes:
        thresholds = traffic_data[col].unique()
        thresholds.sort()
        for threshold in thresholds:
            if len(traffic_data.loc[traffic_data[col] > threshold]) == 0 or \
                    len(traffic_data.loc[traffic_data[col] <= threshold]) == 0:
                continue
            current_gain = calc_information_gain(traffic_data, col, threshold)
            if best_gain[0] is None or current_gain > best_gain[0]:
                best_gain = (current_gain, col, threshold)
    return best_gain[1:]


# class1 = aggressive
# class2 = non-aggressive
def get_entropy(class1, class2):
    """
    calculates the entropy for a particular node
    :param class1: the aggressive drivers in the node
    :param class2: the non-aggressive drivers in the node
    :return: the calculated entropy for that node
    """
    total = class1 + class2
    prob = class1 / total
    if prob == 0:
        return 0
    entropy = prob * math.log2(prob)
    prob = class2 / total
    if prob == 0:
        return 0  # either 0 or 1 indicates completely pure node
    entropy += prob * math.log2(prob)
    entropy = 0 - entropy  # not sure if 'return -entropy' works
    return entropy


def calc_information_gain(traffic_data, attribute, threshold):
    """
    calculates the information gain of a possible split
    given the attribute and threshold to split on
    :param traffic_data: the dataframe containing all traffic data
    :param attribute: the attribute to split on
    :param threshold: the threshold to split on
    :return: the calculated information gain of the given split
    """
    # might have to change depending on sign
    split1 = traffic_data.loc[traffic_data[attribute] > threshold]
    split2 = traffic_data.loc[traffic_data[attribute] <= threshold]
    num_split1 = len(split1)
    num_split2 = len(split2)
    total = num_split1 + num_split2
    entropies = (num_split1 / total) * get_entropy(len(split1.loc[split1['INTENT'] == 2]),
                                                   len(split1.loc[split1['INTENT'] < 2]))
    entropies += (num_split2 / total) * get_entropy(len(split2.loc[split2['INTENT'] == 2]),
                                                    len(split2.loc[split2['INTENT'] < 2]))
    parent_entropy = get_entropy(len(traffic_data.loc[traffic_data['INTENT'] == 2]),
                                 len(traffic_data.loc[traffic_data['INTENT'] < 2]))
    return parent_entropy - entropies


def make_decision_tree(traffic_data, call_depth, attributes_available):
    """
    a recursive function that makes a decision tree in a new python
    file based on the given training data; gets the optimal splits
    and uses them to create the tree
    :param traffic_data: the dataframe containing all traffic data
    :param call_depth: the current depth the recursive function is in
    :param attributes_available: the unused available attributes
    :return: a string containing the decision tree
    """
    # if at max depth or 5 or fewer elements, return average class
    # if tie, will choose aggressive
    # call depth - 1 because we start with 2 indents
    if call_depth-1 == MAX_DEPTH or len(traffic_data) <= LEAST_DATA_POINTS:
        if len(traffic_data.loc[traffic_data['INTENT'] < 2]) > len(traffic_data.loc[traffic_data['INTENT'] == 2]):
            return f"{INDENT * call_depth}intent = {NON_AGGRESSIVE}"
        else:
            return f"{INDENT * call_depth}intent = {AGGRESSIVE}"

    # majority of the node (>90%) is 1 class, return that class
    if len(traffic_data.loc[traffic_data['INTENT'] == 2]) / len(traffic_data) >= 0.88:
        return f"{INDENT * call_depth}intent = {AGGRESSIVE}"
    if len(traffic_data.loc[traffic_data['INTENT'] < 2]) / len(traffic_data) >= 0.88:
        return f"{INDENT * call_depth}intent = {NON_AGGRESSIVE}"

    # if not leaf node, make new left/right split
    best_split = get_best_split(traffic_data, attributes_available)  # gets (attribute, threshold)
    split_gt = traffic_data.loc[traffic_data[best_split[0]] > best_split[1]]
    split_lte = traffic_data.loc[traffic_data[best_split[0]] <= best_split[1]]
    attributes = [x for x in attributes_available if x != best_split[0]]  # remove the attribute that was used
    return (f"{INDENT * call_depth}if row[\"{best_split[0]}\"] > {best_split[1]}:\n"
            f"{make_decision_tree(split_gt, call_depth+1, attributes)}\n"
            f"{INDENT * call_depth}else:\n"
            f"{make_decision_tree(split_lte, call_depth+1, attributes)}")


def make_classifier_file(filename, training_data):
    """
    creates the basis for the classifier file where the tree will be stored
    :param filename: the output file name
    :param training_data: the dataframe containing the training data
    """
    attributes = list(training_data.columns.values)
    attributes.remove('INTENT')
    with open(filename, 'w') as f:
        f.writelines("import sys\n"
                     "import pandas as pd\n\n\n"
                     "def classify(data):\n"
                     f"{INDENT}classifications = list()\n"
                     f"{INDENT}for index, row in data.iterrows():\n"
                     f"{INDENT * 2}intent = 0\n"
                     f"{make_decision_tree(training_data, 2, attributes)}\n"
                     f"{INDENT * 2}classifications.append(intent)\n"
                     f"{INDENT * 2}print(intent)\n"
                     f"{INDENT}csv_out = data.copy()\n"
                     f"{INDENT}csv_out.drop('INTENT', axis=1, inplace=True)\n"
                     f"{INDENT}csv_out['INTENT'] = classifications\n"
                     f"{INDENT}csv_out.to_csv(\"HW_05_Ferioli_Chris__MyClassifications.csv\", index=False)\n\n\n"
                     "def main():\n"
                     "    if len(sys.argv) != 2:\n"
                     "        print(\"invalid args\")\n"
                     "        return\n"
                     "    traffic_data = pd.read_csv(sys.argv[1])\n"
                     "    classify(traffic_data)\n\n\n"
                     "if __name__ == '__main__':\n"
                     "    main()\n")


def main():
    # Read the all the Traffic Data in
    # combined_data = pd.read_csv("cleaned_data.csv")
    combined_data = pd.read_csv("Combined_Data_for_Easy_Numeric_Analysis__v48.csv")

    combined_data.Speed = np.round(combined_data.Speed)
    combined_data.Speed = combined_data.Speed.astype(int)
    # all_traffic_data = clean_data(all_traffic_data)
    # EDA_graphs(all_traffic_data)

    classifier_filename = "HW_05_Classifier_Ferioli_Chris.py"
    make_classifier_file(classifier_filename, combined_data)


if __name__ == '__main__':
    main()
