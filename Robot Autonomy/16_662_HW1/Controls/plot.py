from os import listdir, path
import matplotlib.pyplot as plt
import csv

log_path = '/home/sush/CMU/SLAM-and-Robot-Autonomy/Robot Autonomy/16_662_HW1/Controls'

def main():
    data = {}
    for filename in listdir(log_path):
        if (filename.endswith('.csv')):
            print("processing", filename)
            # y-axis values
            force = []

            # x-axis values
            time = []

            with open(path.join(log_path, filename), 'r') as datafile:
                plot = csv.reader(datafile, delimiter=',')

                count = 0
                for rows in plot:
                    time.append(float(rows[0]))
                    force.append(float(rows[1]))
                    count+=1

            data[filename] = (time, force)

            # plot_costs(time, force, str(filename), 'force (N)')


    plot_together(data['part2_force_vs_time_impedence.csv'], data['part2_force_vs_time_force.csv'], 'part2')
    plot_together(data['part1_force_vs_time_impedence.csv'], data['part1_force_vs_time_force.csv'], 'part1')


def plot_costs(x, y, file_name, y_label):
    # simple graph
    plt.plot(x,y)
    plt.title(file_name)
    plt.xlabel('time (s)')
    plt.ylabel(y_label)
    plt.show()


def plot_together(data1, data2, file_name):
    x1, y1 = data1
    x2, y2 = data2
    plt.plot(x1, y1, 'r', label='Impedence Control')
    plt.plot(x2, y2, 'y', label='Force Control')

    plt.title(file_name)
    plt.xlabel('time (s)')
    plt.ylabel('force (N)')
    plt.legend(loc="upper left")
    plt.show()


if __name__ == '__main__':
    main()