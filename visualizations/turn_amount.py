import csv
import matplotlib.pyplot as plt
import seaborn as sns

dir = "../dataset/glos_cycle_track_wet_clouded/2019-11-12T14_26_47"


def find_total_turn_amount(straight_rows, manual_rows, noise_rows):

    straight_angles = [float(row["angle"]) for row in straight_rows]
    manual_angles = [float(row["angle"]) for row in manual_rows]
    noise_angles = [float(row["angle"]) for row in noise_rows]

    f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

    sns.distplot(straight_angles, kde=False, color="green", ax=axes[0, 0])

    sns.distplot(manual_angles, hist=False, color="grey", ax=axes[0, 1])

    sns.distplot(noise_angles, hist=False, color="blue", ax=axes[1, 0])

    plt.show()
    return


with open(dir + "/driving_log.csv") as data_file:
    reader = csv.DictReader(data_file)
    next(reader, None) # Skip header
    rows = [row for row in reader]


    find_total_turn_amount(rows)

