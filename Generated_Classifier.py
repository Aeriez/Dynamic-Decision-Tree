import sys
import pandas as pd


def classify(data):
    classifications = list()
    for index, row in data.iterrows():
        intent = 0
        if row["HasGlasses"] > 0:
            if row["RoofRack"] > 0:
                if row["SideDents"] > 0:
                    if row["Speed"] > 61:
                        intent = 2
                    else:
                        if row["BumperDamage"] > 0:
                            intent = 2
                        else:
                            intent = 1
                else:
                    if row["Speed"] > 62:
                        if row["BumperDamage"] > 0:
                            intent = 2
                        else:
                            intent = 1
                    else:
                        intent = 1
            else:
                if row["SideDents"] > 0:
                    intent = 2
                else:
                    if row["BumperDamage"] > 0:
                        intent = 2
                    else:
                        if row["Wears_Hat"] > 0:
                            intent = 2
                        else:
                            intent = 2
        else:
            if row["SideDents"] > 0:
                if row["RoofRack"] > 0:
                    if row["Brightness"] > 0:
                        intent = 1
                    else:
                        intent = 2
                else:
                    if row["Wears_Hat"] > 0:
                        intent = 2
                    else:
                        if row["BumperDamage"] > 0:
                            intent = 2
                        else:
                            intent = 2
            else:
                if row["BumperDamage"] > 0:
                    if row["RoofRack"] > 0:
                        if row["Speed"] > 61:
                            intent = 1
                        else:
                            intent = 1
                    else:
                        if row["Speed"] > 62:
                            intent = 2
                        else:
                            intent = 2
                else:
                    intent = 1
        classifications.append(intent)
        print(intent)
    csv_out = data.copy()
    csv_out.drop('INTENT', axis=1, inplace=True)
    csv_out['INTENT'] = classifications
    csv_out.to_csv("HW_05_Ferioli_Chris__MyClassifications.csv", index=False)


def main():
    if len(sys.argv) != 2:
        print("invalid args")
        return
    traffic_data = pd.read_csv(sys.argv[1])
    classify(traffic_data)


if __name__ == '__main__':
    main()
