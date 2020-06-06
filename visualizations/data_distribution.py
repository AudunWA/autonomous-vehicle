import csv
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

dir = "../dataset/glos_cycle_track_wet_clouded/2019-11-12T14_26_47"
straight_dir = "../dataset/glos_cycle_straight_mini/2019-11-14T13:57:39"
manual_noise_dir = "../dataset/glos_cycle_manual_noise_mini/2019-11-14T14:26:44"
tri_noise_dir_1 = "../dataset/glos_cycle_noise_mini/2019-11-14T15:07:15"
tri_noise_dir_2 = "../dataset/glos_cycle_noise_mini/2019-11-14T15:27:08"

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def read_data(dirs):
    rows = []
    for dir in dirs:
        with open(dir + "/driving_log.csv") as f:
            reader = csv.DictReader(f)
            next(reader, None) # Skip header
            rows += [row for row in reader]

    return rows

def get_dist(rows):
    """
        input:
            inputs: dictionary of input data
            targets: dictionary of result data

        return:
            dist: dictionary of distributions for HLC, speed, angle
    """

    dist = {
        "steering": {
            "left": 0,  # More than 20 deg left
            "right": 0,  # More than 20 deg right
            "straight": 0  # Between left and right
        },
    }
    # Steering distribution
    for row in rows:
        angle = float(row["angle"])
        if angle < -0.1:
            dist["steering"]["left"] += 1
        elif angle > 0.1:
            dist["steering"]["right"] += 1
        else:
            dist["steering"]["straight"] += 1

    return dist


def balance_steering_angle(rows, dist):
    """ Balance steer angle such that target fraction is correct """

    # Find the steering with least amount of values
    least_vals = min(dist["steering"]["straight"], dist["steering"]["left"], dist["steering"]["right"])

    rows_after = []

    left_count = 0
    right_count = 0
    forward_count = 0

    for i in range(len(rows)):
        angle = float(rows[i]["angle"])
        is_left = angle < -0.1
        is_right = angle > 0.1

        if is_left and left_count >= least_vals:
            continue

        elif is_right and right_count >= least_vals:
            continue

        elif not is_left and not is_right and forward_count >= least_vals:
            continue

        if is_left:
            left_count += 1
        elif is_right:
            right_count += 1
        else:
            forward_count += 1

        # Keep
        rows_after.append(rows[i])

    return rows_after


def plot_data(dist, ax, title):
    """ Plots distribution of HLC, speed and traffic lights """
    print("Plotting...")
    tot_num = 0

    # Steering
    labels = ["Left", "Straight", "Right"]
    print("Dist: ", dist)
    sizes = [dist["steering"]["left"], dist["steering"]["straight"], dist["steering"]["right"]]

    wedges, texts, autotexts = ax.pie(sizes, autopct='%.1f%%')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title(title)
    ax.legend(wedges, labels, loc="best")
    ax.axis('equal')



straight = read_data([straight_dir])
manual = read_data([manual_noise_dir])
tri = read_data([tri_noise_dir_1, tri_noise_dir_2])

straight_dist = get_dist(straight)
man_dist = get_dist(manual)
tri_dist = get_dist(tri)

fig, axs = plt.subplots(1, 1)
plot_data(tri_dist, axs, "")

plt.show()

# balanced_rows = balance_steering_angle(rows, dist)

# dist_after = get_dist(balanced_rows)

# plot_data(dist_after, "After distribution")



