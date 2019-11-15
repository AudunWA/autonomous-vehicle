import csv
import matplotlib.pyplot as plt
import numpy as np
import math

dir_train = "../dataset/closed_road_eberg/2019-10-23T133745"

fps = 60

grey = "#d5d9dd"
default_alpha = 0.3
active_alpha = 1
def display_hlc(data_file):
    reader = csv.DictReader(data_file)
    next(reader, None) # Skip header

    fig, ax = plt.subplots(1, 1)

    img = None
    left_arrow = None
    forward_arrow = None
    right_arrow = None
    count = 0
    for row in reader:

        count += 1

        if count % 10 != 0: # Skip half the frames (fast forward)
            continue

        angle = float(row["angle"])

        img_path = row["image_path"].split("/")[-1]

        read_image = plt.imread(dir_train + "/images/" + img_path)

        if img is None:
            img = ax.imshow(read_image)
        else:
            img.set_data(read_image)


        center_vertical = len(read_image) / 2
        center_horizontal = len(read_image[0]) / 2

        hlc = int(row["high_level_command"])
        print("HLC", hlc, row)
        if left_arrow is not None:
            left_arrow.remove()
            forward_arrow.remove()
            right_arrow.remove()

        base = center_vertical + 50
        left_alpha = active_alpha if hlc == 0 else default_alpha
        forward_alpha = active_alpha if hlc == 1 else default_alpha
        right_alpha = active_alpha if hlc == 2 else default_alpha

        left_arrow = ax.arrow(center_horizontal - 20, base, -35, 0, color=grey, width=6, alpha=left_alpha)
        forward_arrow = ax.arrow(center_horizontal, base - 20, 0, -35, color=grey, width=6, alpha=forward_alpha)
        right_arrow = ax.arrow(center_horizontal + 20, base, 35, 0, color=grey, width=6, alpha=right_alpha)


        fig.canvas.draw_idle()
        plt.pause(1.0 / fps)



with open(dir_train + "/driving_log.csv") as data_file:
    display_hlc(data_file)

