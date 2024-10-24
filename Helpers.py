import pandas as pd
import csv
import numpy as np


def clean_csv(filename):
    data = pd.read_csv(filename)
    non_aggressive_to_remove = len(data.loc[data['INTENT'] < 2]) - len(data.loc[data['INTENT'] == 2])
    with open(filename, 'r') as fin, open('cleaned_data.csv', 'w', newline='') as fout:

        # define reader and writer objects
        reader = csv.reader(fin, skipinitialspace=True)
        writer = csv.writer(fout, delimiter=',')

        # write headers
        writer.writerow(next(reader))

        # iterate and write rows based on condition
        for i in reader:
            if int(i[-1]) < 2 and non_aggressive_to_remove != 0:
                non_aggressive_to_remove -= 1
            else:
                writer.writerow(i)


def check_attributes_validity(traffic_data):
    columns = list(traffic_data.columns.values)
    columns.remove('INTENT')
    columns_to_remove = list()
    for col in columns:
        least_badness = -1
        thresholds = traffic_data[col].unique()
        thresholds.sort()
        for threshold in thresholds:
            # if target is > threshold
            false_alarm_gt = len(traffic_data.loc[
                                     (traffic_data[col] > threshold) & (traffic_data['INTENT'] < 2)
                                     ])

            false_negative_gt = len(traffic_data.loc[
                                        (traffic_data[col] <= threshold) & (traffic_data['INTENT'] == 2)
                                        ])

            # if target is <= threshold
            false_alarm_lt = len(traffic_data.loc[
                                     (traffic_data[col] <= threshold) & (traffic_data['INTENT'] < 2)
                                     ])

            false_negative_lt = len(traffic_data.loc[
                                        (traffic_data[col] > threshold) & (traffic_data['INTENT'] == 2)
                                        ])

            current_badness_gt = false_alarm_gt + false_negative_gt
            current_badness_lt = false_alarm_lt + false_negative_lt
            if current_badness_gt < current_badness_lt:
                if least_badness == -1 or current_badness_gt < least_badness:
                    least_badness = current_badness_gt
            else:
                if least_badness == -1 or current_badness_lt < least_badness:
                    least_badness = current_badness_lt
        if least_badness == 3207:
            columns_to_remove.append(col)
    return columns_to_remove


def clean_data(filename):
    data = pd.read_csv(filename)
    columns = check_attributes_validity(data)
    data = data.drop(columns=columns)
    pd.DataFrame.to_csv(data, "cleaned_data.csv", index=False)


def get_accuracy(testing_data, classifications):  # we are not given intent, unknown accuracy for now
    testing_data = pd.read_csv(testing_data)
    output_classifications = pd.read_csv(classifications)
    actual = testing_data['INTENT'].tolist()
    classified = output_classifications['INTENT'].tolist()
    num_correct = 0
    true_negative = 0
    false_negative = 0
    true_positive = 0
    false_positive = 0
    for i in range(len(actual)):
        if ((actual[i] < 2 and classified[i] < 2) or
                (actual[i] == 2 and classified[i] == 2)):
            num_correct += 1

        if actual[i] < 2 and classified[i] < 2:
            true_negative += 1
        if actual[i] == 2 and classified[i] < 2:
            false_negative += 1
        if actual[i] == 2 and classified[i] == 2:
            true_positive += 1
        if actual[i] < 2 and classified[i] == 2:
            false_positive += 1
    confusion_matrix = np.array([[true_positive, false_negative], [false_positive, true_negative]])
    print(confusion_matrix)
    return num_correct / len(actual)


def main():
    if True:
        testing_data = "Combined_Data_for_Easy_Numeric_Analysis__v48.csv"
        classifications = "HW_05_Ferioli_Chris__MyClassifications.csv"
        print(f"Accuracy: {get_accuracy(testing_data, classifications)*100}%")

    else:
        clean_csv("Combined_Data_for_Easy_Numeric_Analysis__v48.csv")


if "__main__" == __name__:
    main()
