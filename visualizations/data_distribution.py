import csv
import numpy as np
import matplotlib.pyplot as plt

dir = "../dataset/glos_cycle_track_wet_clouded/2019-11-12T14_26_47"


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


def plot_data(dist, title=""):
    """ Plots distribution of HLC, speed and traffic lights """
    print("Plotting...")
    tot_num = 0

    fig = plt.figure(figsize=(16, 6))
    # Steering
    labels = ["Left", "Straight", "Right"]
    print("Dist: ", dist)
    sizes = [dist["steering"]["left"], dist["steering"]["straight"], dist["steering"]["right"]]

    ax2 = fig.add_subplot(1, 1, 1)
    wedges, texts, autotexts = ax2.pie(sizes, autopct='%.1f%%')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title("Distrubution of steering angle")
    ax2.legend(wedges, labels, loc="best")
    ax2.axis('equal')

    fig.suptitle(title, fontsize=18)
    plt.show()

with open(dir + "/driving_log.csv") as data_file:
    reader = csv.DictReader(data_file)
    next(reader, None) # Skip header
    rows = [row for row in reader]

    dist = get_dist(rows)

    #plot_data(dist, "Before distribution")

    balanced_rows = balance_steering_angle(rows, dist)

    dist_after = get_dist(balanced_rows)

    plot_data(dist_after, "After distribution")



