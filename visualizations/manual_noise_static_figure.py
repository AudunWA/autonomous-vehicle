import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
plt.style.use('seaborn')
import math
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

dir_train = "../dataset/glos_cycle_manual_noise_mini/2019-11-14T14:26:44"
dir_test = "../dataset/glos_cycle_test_data/2019-11-13T14:20:44"

fps = 15

def map_range(value, input_start, input_end, output_start, output_end):
    input_range = input_end - input_start;
    output_range = output_end - output_start;

    return (value - input_start) * output_range / input_range + output_start

def crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def display_actual_driving_affects(data_file):

    reader = csv.DictReader(data_file)
    next(reader, None) # Skip header

    [next(reader) for i in range(7350)] # Skip first 5000 frames

    gs = gridspec.GridSpec(3, 2)
    gs.update(wspace=0.025, hspace=0.025)
    plt.figure()
    ax_1 = plt.subplot(gs[:, 0])
    ax_2_top = plt.subplot(gs[0, 1])
    ax_2_mid = plt.subplot(gs[1, 1])
    ax_2_bot = plt.subplot(gs[2, 1])

    ax_2_top.axis("off")
    ax_2_mid.axis("off")
    ax_2_bot.axis("off")

    seq_len = 100

    rows = [row for row in reader][0:seq_len]
    angle_w_noise = [float(row["angle_w_noise"]) for row in rows]
    angle_noise_only = [float(row["angle_w_noise"]) - float(row["angle"]) for row in rows]
    angle_actual_only = [ float(row["angle"]) for row in rows]

    img_path = rows[0]["image_path"].split("/")[-1]

    read_image = plt.imread(dir_train + "/images/" + img_path)[120:-50] # Crop image

    center_vertical = read_image.shape[0] / 2
    center_horizontal = len(read_image[0]) / 2

    right = read_image.shape[1]
    left = 0

    bot = center_vertical * 2
    y = range(int(bot), 0, -1)

    def plot(axis, label, w_noise, noise_only, actual_only, img_path):
        img_path = img_path.split("/")[-1]

        read_image = plt.imread(dir_train + "/images/" + img_path)[120:-50]  # Crop image

        x = [min(max(center_horizontal + i * -np.average(w_noise), left), right) for i in range(len(y))]
        x_noise = [min(max(center_horizontal + i * -np.average(noise_only), left), right) for i in
                   range(len(y))]
        x_actual = [min(max(center_horizontal + i * -np.average(actual_only), left), right) for i in
                    range(len(y))]

        axis.imshow(read_image)

        axis.plot(x, y, color='green', linewidth=5, label="Output steering")
        axis.text(right - 50, len(y) - 20, label, fontsize='20', color='white')

    plot(ax_2_top, "(a)", angle_w_noise[5:20], angle_noise_only[5:20], angle_actual_only[5:20], rows[12]["image_path"])
    plot(ax_2_mid, "(b)", angle_w_noise[35:50], angle_noise_only[35:50], angle_actual_only[35:50], rows[42]["image_path"])
    plot(ax_2_bot, "(c)", angle_w_noise[65:80], angle_noise_only[65:80], angle_actual_only[65:80], rows[72]["image_path"])

    y = np.arange(0, seq_len / 15, 1 / 15)

    last_ln, = ax_1.plot(y, angle_w_noise, color='green', linewidth=5, label="Output steering")


    ax_1.axvline(12 / 15.0, ymin=0.05, ymax=0.85, ls='--')
    ax_1.axvline(42 / 15.0, ymin=0.05, ymax=0.85, ls='--')
    ax_1.axvline(72 / 15.0, ymin=0.05, ymax=0.85, ls='--')

    ax_1.text(12 / 15.0 - 0.1, 0.80, "(a)", fontsize='20')
    ax_1.text(42 / 15.0 - 0.1, 0.80, "(b)", fontsize='20')
    ax_1.text(72 / 15.0 - 0.1, 0.80, "(c)", fontsize='20')
    ax_1.set_ylabel('Steering')
    ax_1.set_xlabel('Time (s)')
    ax_1.legend()
    plt.show()

with open(dir_train + "/driving_log.csv") as data_file:
    display_actual_driving_affects(data_file)

