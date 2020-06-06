import csv

straight_dir = "../dataset/glos_cycle_straight_mini/2019-11-14T13:57:39"
manual_noise_dir = "../dataset/glos_cycle_manual_noise_mini/2019-11-14T14:26:44"
tri_noise_dir_1 = "../dataset/glos_cycle_noise_mini/2019-11-14T15:07:15"
tri_noise_dir_2 = "../dataset/glos_cycle_noise_mini/2019-11-14T15:27:08"
large_no_sine_dir = "../dataset/glos_cycle_test_short/no_sine_big/run_2"
large_sine_dir = "../dataset/glos_cycle_test_short/sine_big/run_2"


def read_data(dirs):
    rows = []
    for dir in dirs:
        with open(dir + "/driving_log.csv") as f:
            reader = csv.DictReader(f)
            next(reader, None)  # Skip header
            rows += [row for row in reader]

    return rows


def calculate_whiteness(rows):
    """
    Calculates the whiteness of a whole dataset
    :return: The whiteness value
    """
    acc = 0
    for i in range(0, len(rows) - 1):
        acc += (float(rows[i + 1]["angle"]) - float(rows[i]["angle"])) ** 2
    return acc / len(rows)


def calculate_whiteness_accumulated(rows):
    """
    Calculates the whiteness at each time step.
    :return: An array of all accumulated whiteness values
    """
    acc = 0
    whitenesses = []
    for i in range(0, len(rows) - 1):
        acc += (float(rows[i + 1]["angle"]) - float(rows[i]["angle"])) ** 2
        whitenesses.append(acc / len(rows))
        # whitenesses.append(acc / (i + 2))
    return whitenesses


def calculate_whiteness_accumulated_window(rows, window=60):
    """
    Calculates the average whiteness of a sliding window of size [window].
    :return: An array of all averages from sliding window
    """
    acc = 0
    whitenesses = []
    for i in range(0, len(rows) - 1 - window, 1):
        acc_local = 0
        for j in range(i, i + window):
            acc_local += (float(rows[j + 1]["angle"]) - float(rows[j]["angle"])) ** 2
        # whitenesses.append(acc / len(rows))
        whitenesses.append(acc_local / window)
        acc += acc_local
    return whitenesses


straight = calculate_whiteness(read_data([straight_dir]))
manual_noise = calculate_whiteness(read_data([manual_noise_dir]))
tri_noise = calculate_whiteness(read_data([tri_noise_dir_1, tri_noise_dir_2]))
large_no_sine = calculate_whiteness(read_data([large_no_sine_dir])[300:5410])
large_sine = calculate_whiteness(read_data([large_sine_dir])[350:5460])
ys = calculate_whiteness_accumulated_window(read_data([large_no_sine_dir])[300:5510])
ys_sine = calculate_whiteness_accumulated_window(read_data([large_sine_dir])[350:5560])

import matplotlib.pyplot as plt

print("Straight whiteness: ", straight)
print("Manual noise whiteness: ", manual_noise)
print("Triangular whiteness: ", tri_noise)
print("Large no sine whiteness: ", large_no_sine)
print("Large sine whiteness: ", large_sine)

plt.locator_params(axis='x', nbins=20)
plt.plot(range(300, 5510 - 49 - 12), ys)
plt.plot(range(350, 5560 - 100 + 39), ys_sine)
plt.show()
