import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from pyts.classification import TimeSeriesForest


def doClassification(data, data_test):
    X_train = data.drop(['class'], axis=1) #Split into X and y
    y_train = data['class']
    X_test = data_test.drop(['class'], axis=1)  # Split into X and y
    y_test = data_test['class']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier() #Train and test model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


file_contents = []
file_contents_test = []
for i in range(1,11):
    with open('MHEALTHDATASET/mHealth_subject'+str(i)+'.log', 'r') as file: #Read in log file
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if i == 9 or i == 10:
                file_contents_test.append(row)
            else:
                file_contents.append(row)

#Columns and classes gathered from mhealth readme file
columns = ['chest_acceleration_x', 'chest_acceleration_y', 'chest_acceleration_z', 'ecg_1', 'ecg_2',
           'left_ankle_acceleration_x', 'left_ankle_acceleration_y', 'left_ankle_acceleration_z',
           'left_ankle_gyro_x', 'left_ankle_gyro_y', 'left_ankle_gyro_z', 'left_ankle_magnetometer_x',
           'left_ankle_magnetometer_y', 'left_ankle_magnetometer_z', 'right_arm_acceleration_x',
           'right_arm_acceleration_y', 'right_arm_acceleration_z', 'right_arm_gyro_x', 'right_arm_gyro_y',
           'right_arm_gyro_z', 'right_arm_magnetometer_x', 'right_arm_magnetometer_y', 'right_arm_magnetometer_z',
           'class']
classes = {0: 'inbetween', 1: 'standing still', 2: 'sitting and relaxing', 3: 'lying down', 4: 'walking',
           5: 'climbing stairs', 6: 'waist bends forward', 7: 'frontal elevation of arms',
           8: 'knees bending (crouching)', 9: 'cycling', 10: 'jogging', 11: 'running', 12: 'jump front and back'}


data = pd.DataFrame(file_contents, columns=columns)
data = data.astype(float)
data['class'] = data['class'].astype(int)
#arbitrary_start_time = pd.Timestamp('2024-01-01 00:00:00')
#data['timestamp'] = pd.date_range(start=arbitrary_start_time, periods=len(data), freq='20ms') #50 Hz = 20 milliseconds

data = data[data['class'] != 0] #Remove the in-between recordings

data_test = pd.DataFrame(file_contents_test, columns=columns)
data_test = data_test.astype(float)
data_test['class'] = data_test['class'].astype(int)
data_test = data_test[data_test['class'] != 0]
doClassification(data, data_test)



#VISUALIZATION STUFF
# def moving_average_filter(signal, window_size):
#     filter_coeffs = np.ones(window_size) / window_size
#     filtered_signal = np.convolve(signal, filter_coeffs, mode='valid')
#     return filtered_signal
#
# subset = data[data['class'] == 6]
# #subset = subset.head(500)
#
# time_interval = 1 / 50
# subset['chest_acceleration_z'] = subset['chest_acceleration_z'] - subset['chest_acceleration_z'].mean()
# filtered_acceleration = moving_average_filter(subset['chest_acceleration_z'], 100)
# #filtered_acceleration = np.array([1, 2, 3, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8])
# velocity = np.zeros_like(filtered_acceleration)
# for i in range(1, len(velocity)):
#     velocity[i] = velocity[i-1] + filtered_acceleration[i-1]*time_interval
#     #subset.loc[i, 'chest_velocity_z'] = subset.loc[i-1, 'chest_velocity_z'] + subset.loc[i-1, 'chest_acceleration_z']*time_interval
# plt.plot(velocity)
# plt.xticks(rotation=20)
# plt.xlabel('Time')
# plt.ylabel('Chest Movement')
# plt.show()
#
#
#
#
# crouches = data[data['class'] == 8]
#
# dt = 0.02
# velocity = np.cumsum(crouches['chest_acceleration_z'].values)*dt
# position = np.cumsum(velocity)*dt
#

#
# normalized_acceleration_z = crouches['chest_acceleration_z'] - crouches['chest_acceleration_z'].mean()
# filtered = moving_average_filter(normalized_acceleration_z, 10)
# velocity_z = np.cumsum(filtered)*dt
# velocity_z_normalized = velocity_z - velocity_z.mean()
# position_z = np.cumsum(velocity_z_normalized)*dt
# # crouches['chest_velocity_x'] = crouches['chest_acceleration_x'].cumsum()
# # crouches['chest_velocity_y'] = crouches['chest_acceleration_y'].cumsum()
# # crouches['chest_velocity_z'] = crouches['chest_acceleration_z'].cumsum()
# #filtered = moving_average_filter(crouches['chest_acceleration_z'], 10)
#
# plt.plot(data['timestamp'], data['chest_acceleration_z'])
# plt.xticks(rotation=20)
# plt.xlabel('Time')
# plt.ylabel('Chest Movement')
# plt.show()
#
# # fig = plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.plot3D(crouches['chest_acceleration_x'], crouches['chest_acceleration_y'], crouches['chest_acceleration_z'])
# # plt.show()
#



