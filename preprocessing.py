import csv
import numpy as np
from copy import deepcopy as dcp
from motion_mapping import MotionMapping
from marker import marker
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class vicon_traj:
    def __init__(self, filename, datafile):

        self.key1 = None
        self.key2 = ['X', 'Y', 'Z']
        self.data = []
        self.data_dict = {}
        self.n_frame = 0

        count = 0
        with open(filename) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if 2 == count:
                    self.key1 = dcp(row)
                    while '' in self.key1:
                        self.key1.remove('')
                    break
                count += 1

        with open(datafile) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in readCSV:
                self.n_frame += 1
                for i in range(len(row)):
                    if type(row[i]) == str:
                        if row[i] == '':
                            row[i] = row[i-1]
                        else:
                            row[i] = float(row[i])
                            # self.data.append(row)
                    # else:
                    #     self.data.append(row)

                self.data.append(row)

        self.data = np.array(self.data)

        for i in range(len(self.key1)):
            key = self.key1[i].strip('Michelle:')
            # key = self.key1[i].strip('Lo:')
            self.data_dict[key] = self.data[:, 3*i:3*i+3]

    def get_markers(self, t):
        markers = {}
        markers[0] = marker(x=self.data_dict['Lo:RSHO'][t][0], y=self.data_dict['Lo:RSHO'][t][1], z=self.data_dict['Lo:RSHO'][t][2])
        markers[1] = marker(x=self.data_dict['Lo:RELB'][t][0], y=self.data_dict['Lo:RELB'][t][1], z=self.data_dict['Lo:RELB'][t][2])
        markers[2] = marker(x=self.data_dict['Lo:RWRA'][t][0], y=self.data_dict['Lo:RWRA'][t][1], z=self.data_dict['Lo:RWRA'][t][2])
        markers[3] = marker(x=self.data_dict['Lo:RWRB'][t][0], y=self.data_dict['Lo:RWRB'][t][1], z=self.data_dict['Lo:RWRB'][t][2])
        markers[4] = marker(x=self.data_dict['Lo:RFIN'][t][0], y=self.data_dict['Lo:RFIN'][t][1], z=self.data_dict['Lo:RFIN'][t][2])
        markers[13] = marker(x=self.data_dict['Lo:R_MidFinger'][t][0], y=self.data_dict['Lo:R_MidFinger'][t][1], z=self.data_dict['Lo:R_MidFinger'][t][2])
        markers[10] = marker(x=self.data_dict['Lo:R_Thumb'][t][0], y=self.data_dict['Lo:R_Thumb'][t][1], z=self.data_dict['Lo:R_Thumb'][t][2])

        markers[5] = marker(x=self.data_dict['Lo:LSHO'][t][0], y=self.data_dict['Lo:LSHO'][t][1], z=self.data_dict['Lo:LSHO'][t][2])
        markers[6] = marker(x=self.data_dict['Lo:LELB'][t][0], y=self.data_dict['Lo:LELB'][t][1], z=self.data_dict['Lo:LELB'][t][2])
        markers[7] = marker(x=self.data_dict['Lo:LWRA'][t][0], y=self.data_dict['Lo:LWRA'][t][1], z=self.data_dict['Lo:LWRA'][t][2])
        markers[8] = marker(x=self.data_dict['Lo:LWRB'][t][0], y=self.data_dict['Lo:LWRB'][t][1], z=self.data_dict['Lo:LWRB'][t][2])
        markers[9] = marker(x=self.data_dict['Lo:LFIN'][t][0], y=self.data_dict['Lo:LFIN'][t][1], z=self.data_dict['Lo:LFIN'][t][2])
        markers[14] = marker(x=self.data_dict['Lo:L_MidFinger'][t][0], y=self.data_dict['Lo:L_MidFinger'][t][1], z=self.data_dict['Lo:L_MidFinger'][t][2])
        markers[11] = marker(x=self.data_dict['Lo:L_Thumb'][t][0], y=self.data_dict['Lo:L_Thumb'][t][1], z=self.data_dict['Lo:L_Thumb'][t][2])

        markers[12] = marker(x=0.1, y=0.1, z=0.1)
        for i in range(15, 23):
            markers[i] = marker(x=0.1, y=0.1, z=0.1)

        return markers

if __name__ == '__main__':
    example = vicon_traj('Lo03.csv', 'data03.csv')
    model = MotionMapping()
    markers = example.get_markers(0)

    angles = []
    for i in range(example.n_frame):
        markers = example.get_markers(i)
        model.update_vicon_marker(markers)
        model.update_arm_length()
        angles_right = model.cal_angles(model.elbow_right_r, model.wrist_right_r2,
                                        model.wrist_right_r1, model.hand_right_r,
                                        model.index_right_r)
        angles.append(angles_right)

    example2 = vicon_traj('Lo04.csv', 'data04.csv')
    model2 = MotionMapping()
    markers2 = example2.get_markers(0)

    angles2 = []
    for i in range(example2.n_frame):
        markers = example2.get_markers(i)
        model2.update_vicon_marker(markers)
        model2.update_arm_length()
        angles_right = model2.cal_angles(model2.elbow_right_r, model2.wrist_right_r2,
                                        model2.wrist_right_r1, model2.hand_right_r,
                                        model2.index_right_r)
        angles2.append(angles_right)


    fig, ax = plt.subplots()
    ax.plot(angles, 'k--', label='Side grasping')
    ax.plot(angles2, 'k:', label='Top grasping')

    legend = ax.legend(loc='upper right')

    # plt.figure()
    # plt.plot(angles)
    # plt.plot(angles2)
    plt.show()
    # print ('')
