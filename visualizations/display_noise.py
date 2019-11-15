import csv
import matplotlib.pyplot as plt
import numpy as np
import math

dir_train = "../dataset/glos_cycle_track_wet_clouded/2019-11-12T14_26_47"
dir_test = "../dataset/glos_cycle_test_data/2019-11-13T14:20:44"

fps = 15


def display_test_driving(data_file, skip):
    reader = csv.DictReader(data_file)
    next(reader, None) # Skip header
    [next(reader) for i in range(skip)] # Skip first frames

    fig, ax = plt.subplots(1, 1)

    img = None
    last_ln = None
    for row in reader:

        angle = float(row["angle"])

        print("row", row, angle)
        img_path = row["image_path"].split("/")[-1]

        read_image = plt.imread(dir_train + "/images/" + img_path)

        if img is None:
            img = ax.imshow(read_image)
        else:
            img.set_data(read_image)

        center_vertical = len(read_image) / 2
        center_horizontal = len(read_image[0]) / 2

        bot = center_vertical * 2
        y = range(int(bot), 0, -1)

        x = [center_horizontal + i * -angle for i in range(len(y))]

        if last_ln:
            last_ln.remove()
        last_ln, = ax.plot(x, y, color='green', linewidth=5)

        fig.canvas.draw_idle()
        plt.pause(1.0 / fps)


def display_actual_driving_affects(data_file):

    reader = csv.DictReader(data_file)
    next(reader, None) # Skip header

    [next(reader) for i in range(5000)] # Skip first 5000 frames

    fig, ax = plt.subplots(1, 1)
    plt.axis('off')

    img = None
    last_ln = None
    last_noise = None
    last_actual = None
    for row in reader:

        angle_w_noise = float(row["angle_w_noise"])
        angle_noise_only = float(row["angle_w_noise"]) - float(row["angle"])
        angle_actual_only = float(row["angle"])

        img_path = row["image_path"].split("/")[-1]

        print("Angle to motors:", row["angle_w_noise"])
        read_image = plt.imread(dir_train + "/images/" + img_path)

        if img is None:
            img = ax.imshow(read_image)
        else:
            img.set_data(read_image)

        right = read_image.shape[1]
        left = 0
        center_vertical = read_image.shape[0] / 2
        center_horizontal = len(read_image[0]) / 2

        bot = center_vertical * 2
        y = range(int(bot), 0, -1)

        x = [min(max(center_horizontal + i * -angle_w_noise, left), right) for i in range(len(y))]
        x_noise = [min(max(center_horizontal + i * -angle_noise_only, left), right) for i in range(len(y))]
        x_actual = [min(max(center_horizontal + i * -angle_actual_only, left), right) for i in range(len(y))]


        if last_actual:
            last_ln.remove()
            last_noise.remove()
            last_actual.remove()
        last_ln, = ax.plot(x, y, color='green', linewidth=5, label="Output steering")
        last_noise, = ax.plot(x_noise, y, color='grey', label="Noise signal")
        last_actual, = ax.plot(x_actual, y, color='blue', label="Manual signal")

        ax.legend(loc="upper left")

        fig.canvas.draw_idle()
        plt.pause(1.0 / fps)


with open(dir_train + "/driving_log.csv") as data_file:
    display_actual_driving_affects(data_file)

#with open(dir_test + "/driving_log.csv") as data_file:
#    display_test_driving(data_file, 250)


