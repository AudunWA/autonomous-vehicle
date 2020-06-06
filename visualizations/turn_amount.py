import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

dir_triangular_noise = "../dataset/glos_cycle_test_short/triangular_noise"
dir_manual_noise = "../dataset/glos_cycle_test_short/manual_noise"
dir_straight = "../dataset/glos_cycle_test_short/straight"

dir_sine = "../dataset/glos_cycle_test_short/sine_big"
dir_no_sine = "../dataset/glos_cycle_test_short/no_sine_big"


def get_angles(rows):
    return [float(row["angle"]) for row in rows]


def find_total_turn_amount_noise(straight_rows, manual_rows, noise_rows, df):

    straight_angles = [float(row["angle"]) for row in straight_rows]
    manual_angles = [float(row["angle"]) for row in manual_rows]
    noise_angles = [float(row["angle"]) for row in noise_rows]

    f, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

    sns.distplot(straight_angles, label="No noise", ax=axes[0])
    axes[0].set_title("No noise", loc="right")
    sns.distplot(manual_angles, label="Manual noise", ax=axes[1])
    axes[1].set_title("Manual noise", loc="right")
    sns.distplot(noise_angles, label="Triangular noise", ax=axes[2])
    axes[2].set_title("Triangular noise", loc="right")

    plt.xlim(-1, 1)
    f.text(0.5, 0.04, 'Steering signal from left (-1) to right (1).', ha='center', fontsize=14)
    f.text(0.04, 0.5, 'Number of occurrences', va='center', rotation='vertical', fontsize=14)

    f.text(0.8, 0.25, "μ = " + str(round(df["Triangular"].mean(), 4)), fontsize=14)
    f.text(0.8, 0.28, "σ = " + str(round(df["Triangular"].std(), 4)), fontsize=14)

    f.text(0.8, 0.52, "μ = " + str(round(df["Manual"].mean(), 4)), fontsize=14)
    f.text(0.8, 0.55, "σ = " + str(round(df["Manual"].std(), 4)), fontsize=14)

    f.text(0.8, 0.8, "μ = " + str(round(df["Straight"].mean(), 4)), fontsize=14)
    f.text(0.8, 0.83, "σ = " + str(round(df["Straight"].std(), 4)), fontsize=14)

    f.suptitle('Distribution of steering signal by dataset', fontsize=20)

def find_total_turn_amount_sine(sine_rows, no_sine_rows, df):

    sine_angles = [float(row["angle"]) for row in sine_rows]
    no_sine_angles = [float(row["angle"]) for row in no_sine_rows]

    f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    sns.distplot(sine_angles, label="Sine", ax=axes[0])
    axes[0].set_title("Sine", loc="right")
    sns.distplot(no_sine_angles, label="No sine", ax=axes[1])
    axes[1].set_title("No sine", loc="right")

    plt.xlim(-1, 1)
    f.text(0.5, 0.04, 'Steering signal from left (-1) to right (1).', ha='center', fontsize=14)
    f.text(0.04, 0.5, 'Number of occurrences', va='center', rotation='vertical', fontsize=14)

    f.text(0.8, 0.7, "μ = " + str(round(df["Sine big"].mean(), 4)), fontsize=14)
    f.text(0.8, 0.73, "σ = " + str(round(df["Sine big"].std(), 4)), fontsize=14)

    f.text(0.8, 0.3, "μ = " + str(round(df["No sine big"].mean(), 4)), fontsize=14)
    f.text(0.8, 0.33, "σ = " + str(round(df["No sine big"].std(), 4)), fontsize=14)

    f.suptitle('Distribution of steering signal by steering representation', fontsize=20)


def read_rows(dir):
    rows = []
    for filename in os.listdir(dir):
        print("path", dir + filename + "/data.csv")
        reader = csv.DictReader(open(dir + "/" + filename + "/data.csv"))
        next(reader, None)  # Skip header
        for row in reader:
            rows.append(row)

    return rows


tri_rows = read_rows(dir_triangular_noise)

man_rows = read_rows(dir_manual_noise)

str_rows = read_rows(dir_straight)

sine_rows = read_rows(dir_sine)

no_sine_rows = read_rows(dir_no_sine)

pd.set_option('display.max_columns', 10)

df = pd.DataFrame(
    list(zip(get_angles(str_rows), get_angles(man_rows), get_angles(tri_rows), get_angles(sine_rows), get_angles(no_sine_rows))),
    columns=["Straight", "Manual", "Triangular", "Sine big", "No sine big"]
)

print(df.describe(include="all"))

find_total_turn_amount_noise(str_rows, man_rows, tri_rows, df)

find_total_turn_amount_sine(sine_rows, no_sine_rows, df)

plt.show()
