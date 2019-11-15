
import csv
import matplotlib.pyplot as plt
import time
import cv2
import seaborn as sns

clock_diff = 10973766144

def to_ms(nano):
    return nano / 1000 / 1000

def calculate_average_time_offset(rows):
    total_offset_abs = 0
    for row in rows:
        total_offset_abs += abs(int(row["ackerman_timestamp"]) + clock_diff - int(row["img_timestamp"]))

    return to_ms(total_offset_abs / len(rows))

def calculate_fps(rows):
    first_timestamp = int(rows[0]["img_timestamp"])
    last_timestamp = int(rows[-1]["img_timestamp"])

    duration = to_ms(last_timestamp - first_timestamp) / 1000

    return 1.0 / (duration / len(rows))

def fps_distribution(rows):
    timestamp = int(rows[0]["img_timestamp"])
    distribution = []
    for row in rows[1:-1]:
        curr = int(row["img_timestamp"])
        duration = to_ms(curr - timestamp) / 1000
        timestamp = curr
        distribution.append(1.0 / duration)

    return distribution

dir = "../dataset/glos_cycle_track_wet_clouded/2019-11-12T14_26_47"


def get_dataset_summary():
    with open(dir + "/driving_log.csv") as data_file:
        reader = csv.DictReader(data_file)
        next(reader, None) # Skip header
        rows = [row for row in reader]
        total_offset = calculate_average_time_offset(rows)
        fps = calculate_fps(rows)
        fps_dist = fps_distribution(rows)
        print("Milliseconds offset of dataset:", total_offset)
        print("Frames per seconds:            ", fps)

        sns.distplot(fps_dist)
        print(fps_dist[0:10])
        plt.show()



def display_training_data():
    with open(dir + "/driving_log.csv") as data_file:
        reader = csv.DictReader(data_file)
        next(reader, None) # Skip header
        plt.axis("off")
        fig, ax = plt.subplots(1, 1)

        img = None
        for row in reader:
            img_path = row["image_path"].split("/")[-1]

            print("Angle to motors:", row["angle_w_noise"])
            read_image = plt.imread(dir + "/images/" + img_path)

            if img is None:
                img = ax.imshow(read_image)
            else:
                img.set_data(read_image)

            fig.canvas.draw_idle()
            plt.pause(.5)


get_dataset_summary()
#display_training_data()